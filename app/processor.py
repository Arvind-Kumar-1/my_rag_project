# app/processor.py
import fitz
import faiss
import numpy as np
import requests
import ollama
import json
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from pathlib import Path
import docx
from email.parser import Parser
import time


class RAGProcessor:
    def __init__(self):
        """Initializes the processor, loading the model once."""
        print("Initializing RAGProcessor and loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 512
        self.overlap = 50
        print("Embedding model loaded successfully.")



    def _download_file(self, url: str) -> bytes:
        """Download file from URL with proper error handling."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading file from {url}: {e}")
            raise

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            # Save temporarily and read with python-docx
            temp_path = Path("temp_doc.docx")
            with open(temp_path, "wb") as f:
                f.write(content)
            
            doc = docx.Document(temp_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Clean up
            temp_path.unlink()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""

    def _extract_text_from_email(self, content: bytes) -> str:
        """Extract text from email content."""
        try:
            msg = Parser().parsestr(content.decode('utf-8', errors='ignore'))
            if msg.is_multipart():
                text = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                return text.strip()
            else:
                return msg.get_payload(decode=True).decode('utf-8', errors='ignore').strip()
        except Exception as e:
            print(f"Error extracting text from email: {e}")
            return ""

    def _detect_file_type(self, url: str, content: bytes) -> str:
        """Detect file type from URL and content."""
        url_lower = url.lower()
        if url_lower.endswith('.pdf'):
            return 'pdf'
        elif url_lower.endswith('.docx'):
            return 'docx'
        elif url_lower.endswith('.doc'):
            return 'doc'
        elif any(ext in url_lower for ext in ['.eml', '.msg']):
            return 'email'
        
        # Check content headers
        if content[:4] == b'%PDF':
            return 'pdf'
        elif content[:4] == b'PK\x03\x04':  # ZIP signature (DOCX is ZIP)
            return 'docx'
        
        return 'text'

    def _smart_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Create intelligent chunks with metadata."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_metadata = {"type": "content", "section": None}
        
        for i, paragraph in enumerate(paragraphs):
            # Check if this looks like a section header
            is_header = (
                len(paragraph) < 100 and 
                (paragraph.isupper() or 
                 re.match(r'^[0-9]+\.', paragraph) or
                 re.match(r'^[A-Z][A-Z\s]+:?$', paragraph))
            )
            
            if is_header:
                # Save current chunk if it exists
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": current_metadata.copy(),
                        "chunk_id": len(chunks)
                    })
                
                # Start new section
                current_metadata["section"] = paragraph
                current_chunk = paragraph + "\n\n"
            else:
                # Add to current chunk
                if len(current_chunk) + len(paragraph) > self.chunk_size:
                    # Save current chunk
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": current_metadata.copy(),
                            "chunk_id": len(chunks)
                        })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                    current_chunk = overlap_text + paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": current_metadata.copy(),
                "chunk_id": len(chunks)
            })
        
        return chunks

    def _extract_query_intent(self, question: str) -> Dict[str, Any]:
        """Analyze query to understand intent and requirements."""
        question_lower = question.lower()
        
        intent = {
            "type": "general",
            "requires_specific_clause": False,
            "temporal": False,
            "conditional": False,
            "comparison": False,
            "keywords": []
        }
        
        # Detect specific patterns
        if any(word in question_lower for word in ['cover', 'covered', 'coverage', 'includes', 'exclude']):
            intent["type"] = "coverage"
            intent["requires_specific_clause"] = True
        
        if any(word in question_lower for word in ['condition', 'requirement', 'criteria', 'must', 'shall']):
            intent["type"] = "conditions"
            intent["requires_specific_clause"] = True
        
        if any(word in question_lower for word in ['when', 'if', 'unless', 'provided that']):
            intent["conditional"] = True
        
        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            intent["comparison"] = True
        
        # Extract key terms
        important_words = re.findall(r'\b[A-Za-z]{3,}\b', question)
        intent["keywords"] = [word.lower() for word in important_words 
                            if word.lower() not in ['the', 'and', 'for', 'are', 'this', 'that', 'what', 'how', 'does']]
        
        return intent

    def _enhanced_retrieval(self, question: str, chunks: List[Dict], embeddings: np.ndarray, index: faiss.Index) -> List[Dict]:
        """Enhanced retrieval with query understanding."""
        query_intent = self._extract_query_intent(question)
        
        # Standard semantic search
        question_embedding = self.embedding_model.encode([question])
        k = min(10, len(chunks))  # Get more candidates initially
        scores, indices = index.search(np.array(question_embedding, dtype=np.float32), k)
        
        # Score and rank chunks
        scored_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = chunks[idx].copy()
            chunk["similarity_score"] = float(1 / (1 + score))  # Convert L2 distance to similarity
            
            # Keyword matching bonus
            keyword_score = 0
            chunk_text_lower = chunk["text"].lower()
            for keyword in query_intent["keywords"]:
                if keyword in chunk_text_lower:
                    keyword_score += 1
            
            chunk["keyword_score"] = keyword_score
            chunk["combined_score"] = chunk["similarity_score"] + (keyword_score * 0.1)
            
            scored_chunks.append(chunk)
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top 5 with diverse content
        final_chunks = []
        used_sections = set()
        
        for chunk in scored_chunks:
            section = chunk["metadata"].get("section", "general")
            if len(final_chunks) < 5 and (section not in used_sections or len(final_chunks) < 3):
                final_chunks.append(chunk)
                used_sections.add(section)
        
        return final_chunks[:5]

    def _generate_structured_answer(self, question: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer with explainability and structured output."""
        # Prepare context
        context_parts = []
        source_info = []
        
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[Source {i+1}] {chunk['text']}")
            source_info.append({
                "source_id": i+1,
                "section": chunk["metadata"].get("section", "Document content"),
                "similarity_score": round(chunk["similarity_score"], 3),
                "chunk_id": chunk["chunk_id"]
            })
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with structure requirements
        prompt = f"""Based ONLY on the provided context sources, answer the following question with precision and clarity.

CONTEXT SOURCES:
{context}

QUESTION: {question}

Please provide your response in the following format:
1. DIRECT ANSWER: [Provide a clear, concise answer]
2. SUPPORTING EVIDENCE: [Quote specific text from the sources that support your answer]
3. CONDITIONS/LIMITATIONS: [Any conditions, limitations, or exceptions mentioned]
4. CONFIDENCE: [High/Medium/Low based on how well the context addresses the question]

If the context doesn't contain enough information to answer the question completely, clearly state what information is missing.

RESPONSE:"""

        try:
            start_time = time.time()
            response = ollama.chat(
                model='llama3', 
                messages=[{'role': 'user', 'content': prompt}], 
                options={'temperature': 0.1, 'top_p': 0.9}
            )
            generation_time = time.time() - start_time
            
            raw_answer = response['message']['content'].strip()
            
            # Parse structured response
            parsed_answer = self._parse_structured_response(raw_answer)
            
            return {
                "answer": parsed_answer.get("direct_answer", raw_answer),
                "evidence": parsed_answer.get("supporting_evidence", ""),
                "conditions": parsed_answer.get("conditions", ""),
                "confidence": parsed_answer.get("confidence", "Medium"),
                "sources": source_info,
                "generation_time": round(generation_time, 2),
                "raw_response": raw_answer
            }
            
        except Exception as e:
            print(f"Error generating answer for '{question}': {e}")
            return {
                "answer": "Error generating answer from the language model.",
                "evidence": "",
                "conditions": "",
                "confidence": "Low",
                "sources": source_info,
                "generation_time": 0,
                "error": str(e)
            }

    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """Parse the structured response from the LLM."""
        parsed = {}
        
        # Extract sections using regex
        sections = {
            "direct_answer": r"(?:1\.\s*DIRECT ANSWER:?\s*)(.*?)(?=(?:\n2\.|\n\nQUESTION|\Z))",
            "supporting_evidence": r"(?:2\.\s*SUPPORTING EVIDENCE:?\s*)(.*?)(?=(?:\n3\.|\n\nQUESTION|\Z))",
            "conditions": r"(?:3\.\s*CONDITIONS/LIMITATIONS:?\s*)(.*?)(?=(?:\n4\.|\n\nQUESTION|\Z))",
            "confidence": r"(?:4\.\s*CONFIDENCE:?\s*)(.*?)(?=(?:\n|\Z))"
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                parsed[key] = match.group(1).strip()
        
        # Fallback: use entire response as direct answer if parsing fails
        if not parsed.get("direct_answer"):
            parsed["direct_answer"] = response
        
        return parsed

    def process_document_and_questions(self, doc_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline - returns simple string answers for API compatibility."""
        try:
            # Download and process document
            content = self._download_file(doc_url)
            file_type = self._detect_file_type(doc_url, content)
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = self._extract_text_from_pdf(content)
            elif file_type == 'docx':
                text = self._extract_text_from_docx(content)
            elif file_type == 'email':
                text = self._extract_text_from_email(content)
            else:
                text = content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                return ["Failed to extract text from the document." for _ in questions]
            
            # Create intelligent chunks
            chunks = self._smart_chunk_text(text)
            if not chunks:
                return ["No content chunks created from the document." for _ in questions]
            
            # Create embeddings and index
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            index.add(np.array(chunk_embeddings, dtype=np.float32))
            
            # Process each question
            final_answers = []
            for question in questions:
                relevant_chunks = self._enhanced_retrieval(question, chunks, chunk_embeddings, index)
                structured_result = self._generate_structured_answer(question, relevant_chunks)
                
                # For API compatibility, return just the answer string
                # In a full implementation, you'd return the structured result
                final_answers.append(structured_result["answer"])
            
            return final_answers
            
        except Exception as e:
            print(f"Error in document processing pipeline: {e}")
            return [f"Error processing document: {str(e)}" for _ in questions]

    def process_document_and_questions_detailed(self, doc_url: str, questions: List[str]) -> List[Dict[str, Any]]:
        """Enhanced version that returns detailed structured responses."""
        try:
            # Download and process document
            content = self._download_file(doc_url)
            file_type = self._detect_file_type(doc_url, content)
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = self._extract_text_from_pdf(content)
            elif file_type == 'docx':
                text = self._extract_text_from_docx(content)
            elif file_type == 'email':
                text = self._extract_text_from_email(content)
            else:
                text = content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                return [{"answer": "Failed to extract text from the document.", "error": "No text extracted"} for _ in questions]
            
            # Create intelligent chunks
            chunks = self._smart_chunk_text(text)
            if not chunks:
                return [{"answer": "No content chunks created from the document.", "error": "No chunks created"} for _ in questions]
            
            # Create embeddings and index
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
            index.add(np.array(chunk_embeddings, dtype=np.float32))
            
            # Process each question
            detailed_results = []
            for question in questions:
                relevant_chunks = self._enhanced_retrieval(question, chunks, chunk_embeddings, index)
                structured_result = self._generate_structured_answer(question, relevant_chunks)
                detailed_results.append(structured_result)
            
            return detailed_results
            
        except Exception as e:
            print(f"Error in document processing pipeline: {e}")
            # Clean up memory after errors too
            self.cleanup_memory()
            return [{"answer": f"Error processing document: {str(e)}", "error": str(e)} for _ in questions]


def cleanup_memory(self):
    """Clean up memory resources if needed"""
    try:
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any cached embeddings if stored
        if hasattr(self, '_cached_embeddings'):
            delattr(self, '_cached_embeddings')
            
    except Exception as e:
        print(f"Warning: Error during memory cleanup: {e}")
        pass