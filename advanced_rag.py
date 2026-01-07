import os
import asyncio
import nest_asyncio
import logging
import traceback
import warnings
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LlamaParse imports
from llama_parse import LlamaParse
from langchain_huggingface import HuggingFaceEmbeddings

# Existing imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message=".*parsing_instruction.*")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("llama_parse").setLevel(logging.WARNING)

# Nest asyncio to allow nested event loops
nest_asyncio.apply()

class RAGChatbot:
    def __init__(self):
        self.setup_logging()
        self.logger.info("Initializing RAGChatbot")
        load_dotenv()
        self.qa_chain = None
        self.vectorstore = None
        # Updated for local VS Code setup
        self.file_path = 'data'  # Changed from '/content/drive/MyDrive/data'
        self.upload_dir = 'data'  # Added for file uploads
        self.processed_files = set()

        # Chat history configuration
        self.chat_history_dir = 'chat_history'
        if not os.path.exists(self.chat_history_dir):
            os.makedirs(self.chat_history_dir)
        self.chat_history_file = os.path.join(self.chat_history_dir, 'chat_history.json')

        # Supported file extensions for LlamaParse
        self.supported_extensions = [
            '.pdf', '.doc', '.docx', '.txt', '.csv', '.xlsx', '.xls',
            '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.gif', '.bmp',
            '.html', '.htm', '.rtf'
        ]

    def setup_logging(self):
        """Set up logging with both file and console output"""
        if not os.path.exists('logs'):
            os.makedirs('logs')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def verify_api_keys(self):
        """Verify GROQ and Llama Parse API keys"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        llama_parse_api_key = os.getenv("LLAMA_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ API key not found in environment variables. Please set GROQ_API_KEY.")

        if not llama_parse_api_key:
            raise ValueError("Llama Parse API key not found in environment variables. Please set LLAMA_API_KEY.")

        self.logger.info("API keys verified")
        return True

    async def parse_document_async(self, file_path, filename):
        """
        Asynchronously parse a single document
        """
        # Check if file extension is supported
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.supported_extensions:
            self.logger.info(f"Skipping unsupported file type: {filename}")
            return []

        try:
            # Initialize LlamaParse
            llama_parser = LlamaParse(
                api_key=os.getenv("LLAMA_API_KEY"),
                num_workers=4,
                verbose=False
            )

            # Use async method for parsing
            parsed_results = await llama_parser.aload_data(file_path)

            # Convert to documents
            documents = []
            for result in parsed_results:
                # Safely extract text, default to empty string if not available
                text = result.text if hasattr(result, 'text') else ''

                document = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "file_type": file_ext.lstrip('.'),
                    }
                )
                documents.append(document)

            self.logger.info(f"Successfully parsed document: {filename}")
            return documents

        except Exception as e:
            self.logger.error(f"Error parsing document {filename}: {str(e)}")
            return []

    def load_documents_with_llamaparse(self):
        """
        Load and parse documents using LlamaParse
        """
        documents = []

        try:
            # Check if data directory exists
            if not os.path.exists(self.file_path):
                self.logger.warning(f"Data directory '{self.file_path}' not found. Creating it...")
                os.makedirs(self.file_path)
                return []

            # Get list of files, excluding hidden or system files
            files = [
                f for f in os.listdir(self.file_path)
                if not f.startswith('.')
                and not f.endswith(('.ipynb_checkpoints', '.DS_Store'))
                and os.path.splitext(f)[1].lower() in self.supported_extensions
            ]

            if not files:
                self.logger.warning(f"No supported files found in '{self.file_path}' directory.")
                return []

            # Run async parsing
            loop = asyncio.get_event_loop()
            parsing_tasks = [
                self.parse_document_async(
                    os.path.join(self.file_path, filename),
                    filename
                )
                for filename in files
            ]

            # Execute all parsing tasks
            parsed_results = loop.run_until_complete(asyncio.gather(*parsing_tasks))

            # Flatten results
            for result in parsed_results:
                documents.extend(result)

            self.logger.info(f"Total documents parsed: {len(documents)}")
            return documents

        except Exception as e:
            self.logger.error(f"Error in document loading: {str(e)}")
            return []

    def process_document(self, file_path: str):
        """Process a single uploaded document"""
        try:
            filename = os.path.basename(file_path)
            loop = asyncio.get_event_loop()
            documents = loop.run_until_complete(
                self.parse_document_async(file_path, filename)
            )
            
            if documents and self.vectorstore:
                # Add to existing vectorstore
                text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
                text_chunks = text_splitter.split_documents(documents)
                self.vectorstore.add_documents(text_chunks)
                self.logger.info(f"Added {len(text_chunks)} chunks from {filename}")
            
            return documents
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            return []

    def initialize(self):
        """Initialize the chatbot components"""
        try:
            self.logger.info("Starting initialization...")

            # Verify API keys
            self.verify_api_keys()

            # Create necessary directories
            for directory in ['logs', 'doc_db', 'data']:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    self.logger.info(f"Created directory: {directory}")

            # Load documents using LlamaParse
            self.logger.info("Loading files...")

            documents = self.load_documents_with_llamaparse()

            if not documents:
                self.logger.warning("No documents were loaded. Using empty document set.")
                # Create a dummy document to prevent complete failure
                documents = [Document(page_content="No documents found", metadata={"source": "system"})]

            self.logger.info(f"Loaded {len(documents)} documents.")

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
            text_chunks = text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(text_chunks)} text chunks.")

            # Initialize embeddings
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                cache_folder="./model_cache"
            )

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=text_chunks,
                embedding=embedding,
                persist_directory="doc_db"
            )

            # Initialize the language model (Groq)
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )

            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
            )
            self.logger.info("Initialization successful")
            return True

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}\n{traceback.format_exc()}")
            return False

    def ask_question(self, query):
        """Process a question with enhanced prompting and return an accurate response"""
        try:
            if not self.qa_chain:
                if not self.initialize():
                    return {
                        "result": "Error: Could not initialize the chatbot. Please check the logs.",
                        "source_documents": []
                    }

            # Improve query formulation
            refined_query = f"""
            You are an expert assistant for document Q&A.
            Answer the following question accurately based on retrieved documents:

            Question: {query}

            Provide a well-structured, factual response.
            If the information is not available in the documents, say so clearly rather than making assumptions.
            """

            response = self.qa_chain.invoke({"query": refined_query})

            # Check if response is valid
            if not response or "result" not in response:
                return {
                    "result": "I couldn't find a relevant answer. Please refine your question.",
                    "source_documents": []
                }

            return response

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {
                "result": f"Error processing question: {str(e)}",
                "source_documents": []
            }

    # Chat History Management Functions
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Retrieve the chat history"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving chat history: {str(e)}")
            return []

    def get_chat_by_id(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chat by ID"""
        chats = self.get_chat_history()
        for chat in chats:
            if chat.get("id") == chat_id:
                return chat
        return None

    def update_chat_history(self, chat_id: str, title: str, messages: List[Dict[str, Any]]) -> bool:
        """Update the chat history"""
        try:
            chats = self.get_chat_history()

            # Check if chat exists
            chat_exists = False
            for i, chat in enumerate(chats):
                if chat.get("id") == chat_id:
                    chats[i] = {
                        "id": chat_id,
                        "title": title,
                        "messages": messages,
                        "updated_at": self._get_timestamp()
                    }
                    chat_exists = True
                    break

            # Add new chat if it doesn't exist
            if not chat_exists:
                chats.append({
                    "id": chat_id,
                    "title": title,
                    "messages": messages,
                    "created_at": self._get_timestamp(),
                    "updated_at": self._get_timestamp()
                })

            # Save updated chat history
            with open(self.chat_history_file, 'w') as f:
                json.dump(chats, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error updating chat history: {str(e)}")
            return False

    def _get_timestamp(self):
        """Generate a timestamp string"""
        return datetime.now().isoformat()

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat by ID"""
        try:
            chats = self.get_chat_history()
            updated_chats = [chat for chat in chats if chat.get("id") != chat_id]

            with open(self.chat_history_file, 'w') as f:
                json.dump(updated_chats, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error deleting chat: {str(e)}")
            return False

    def suggest_title(self, question: str, answer: str) -> str:
        """Generate a title for a new chat"""
        try:
            # Simple title generation (first 5 words of question)
            words = question.split()[:5]
            title = " ".join(words)

            # Add ellipsis if truncated
            if len(words) < len(question.split()):
                title += "..."

            return title

        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            return "New Chat"

    def reset(self) -> bool:
        """Reset the chatbot state"""
        try:
            # Close vectorstore properly
            if self.vectorstore:
                try:
                    self.vectorstore.persist()  # Save any unsaved changes
                    self.vectorstore = None
                except Exception as e:
                    self.logger.warning(f"Error persisting vectorstore: {str(e)}")

            self.qa_chain = None
            self.processed_files = set()
            self.logger.info("Chatbot reset successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting chatbot: {str(e)}")
            return False

    def generate_chat_id(self):
        """Generate a unique chat ID"""
        return f"chat_{uuid.uuid4().hex[:8]}"

    def main(self):
        """Main execution method with conversational interface and chat history support"""
        self.logger.info("Starting the chatbot process...")

        try:
            # Initialize the QA chain
            if not self.initialize():
                print("🤖: Error initializing the chatbot. Please check the logs.")
                return

            print("\n🤖: Hello! I'm your Document Assistant. I can help answer questions about your documents.")
            print("Type 'history' to view past chats, 'load [chat_id]' to continue a chat,")
            print("'delete [chat_id]' to remove a chat, or 'exit' to quit.")

            # Initialize current chat
            chat_id = self.generate_chat_id()
            chat_title = "New Chat"
            chat_messages = []

            while True:
                # Get user input
                user_input = input("\n👤: ").strip()

                # Handle empty input
                if not user_input:
                    print("🤖: Please ask a question or type a command.")
                    continue

                # Handle commands
                if user_input.lower() == 'exit':
                    print("\n🤖: It was great chatting with you! Have a nice day! 👋")
                    break

                elif user_input.lower() == 'history':
                    chats = self.get_chat_history()
                    if not chats:
                        print("🤖: No chat history found.")
                    else:
                        print("\n📚 Chat History:")
                        for chat in chats:
                            print(f"  • ID: {chat['id']} - {chat['title']} ({chat['updated_at']})")
                    continue

                elif user_input.lower() == 'reset':
                    if self.reset():
                        print("🤖: Chatbot has been reset successfully.")
                        chat_id = self.generate_chat_id()
                        chat_title = "New Chat"
                        chat_messages = []
                    else:
                        print("🤖: Error resetting the chatbot. Please check the logs.")
                    continue

                elif user_input.lower().startswith('load '):
                    try:
                        load_chat_id = user_input[5:].strip()
                        loaded_chat = self.get_chat_by_id(load_chat_id)
                        if loaded_chat:
                            chat_id = loaded_chat['id']
                            chat_title = loaded_chat['title']
                            chat_messages = loaded_chat['messages']

                            print(f"🤖: Loaded chat: {chat_title}")
                            # Print last 3 messages for context
                            if chat_messages:
                                print("\n--- Recent Messages ---")
                                for msg in chat_messages[-3:]:
                                    role = "👤" if msg['role'] == 'user' else "🤖"
                                    print(f"{role}: {msg['content']}")
                        else:
                            print(f"🤖: Chat with ID {load_chat_id} not found.")
                        continue
                    except Exception as e:
                        print(f"🤖: Error loading chat: {str(e)}")
                        continue

                elif user_input.lower().startswith('delete '):
                    try:
                        delete_chat_id = user_input[7:].strip()
                        if self.delete_chat(delete_chat_id):
                            print(f"🤖: Chat {delete_chat_id} deleted successfully.")
                            # If we deleted the current chat, create a new one
                            if delete_chat_id == chat_id:
                                chat_id = self.generate_chat_id()
                                chat_title = "New Chat"
                                chat_messages = []
                        else:
                            print(f"🤖: Failed to delete chat {delete_chat_id}.")
                        continue
                    except Exception as e:
                        print(f"🤖: Error deleting chat: {str(e)}")
                        continue

                # Process regular questions
                try:
                    # Add user message to history
                    chat_messages.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": self._get_timestamp()
                    })

                    # Process the question
                    print("🤖: Processing your question...")
                    answer = self.ask_question(user_input)

                    # Extract response text and sources
                    if isinstance(answer, dict):
                        response_text = answer.get("result", "I couldn't find an answer to that question.")
                        sources = answer.get("source_documents", [])

                        # Format source information
                        source_text = ""
                        if sources:
                            unique_sources = set()
                            source_text = "\n📚 Sources:\n"
                            for doc in sources:
                                source = doc.metadata.get('source', 'Unknown source')
                                if source not in unique_sources:
                                    unique_sources.add(source)
                                    source_text += f"   • {source}\n"

                        # Print the main response
                        print(f"\n🤖: {response_text}")
                        if sources:
                            print(source_text)

                        # Add bot message to history
                        chat_messages.append({
                            "role": "assistant",
                            "content": response_text + (f"\n{source_text}" if source_text else ""),
                            "timestamp": self._get_timestamp()
                        })
                    else:
                        print(f"\n🤖: {answer}")
                        # Add bot message to history
                        chat_messages.append({
                            "role": "assistant",
                            "content": str(answer),
                            "timestamp": self._get_timestamp()
                        })

                    # Update chat title if this is the first message
                    if len(chat_messages) == 2:  # After first Q&A pair
                        chat_title = self.suggest_title(user_input, response_text)

                    # Save chat to history
                    self.update_chat_history(chat_id, chat_title, chat_messages)

                except Exception as e:
                    self.logger.error(f"Error processing question: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    print("\n🤖: I encountered an error while processing your question. Please try again or check the logs for details.")
                    continue

        except KeyboardInterrupt:
            print("\n🤖: Chatbot session interrupted. Goodbye!")
        except Exception as e:
            self.logger.error(f"Critical error in main execution: {str(e)}")
            self.logger.debug(traceback.format_exc())
            print("\n🤖: I've encountered a critical error and need to shut down. Please check the logs for details.")
            print(f"\n❌ Error: {str(e)}")

def main():
    """Standalone execution function"""
    chatbot = RAGChatbot()
    chatbot.main()

if __name__ == "__main__":
    main()