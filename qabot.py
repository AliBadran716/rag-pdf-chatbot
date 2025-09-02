from ibm_watsonx_ai.foundation_models import ModelInference  # Importing the ModelInference class for model inference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # Importing generation parameters for text
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames  # Importing embedding parameters for text
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # Importing Watsonx LLM and embeddings for language processing
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing a text splitter for document processing
from langchain_community.vectorstores import Chroma  # Importing Chroma for vector storage
from langchain_community.document_loaders import PyPDFLoader  # Importing PDF loader for document handling
from langchain.chains import RetrievalQA, ConversationalRetrievalChain  # Importing chains for QA and conversational retrieval
from langchain.memory import ConversationBufferMemory  # Importing memory management for conversation history

from cred import credentials, model_id, credential_params  # Importing credentials and model parameters from a separate module

import gradio as gr  # Importing Gradio for creating a web interface

def get_llm():
    """
    Initialize and configure a Watson x LLM instance with specific parameters.
    This function creates a Watson x LLM with predefined settings for token generation
    and temperature to control response randomness.
    Returns:
        WatsonxLLM: A configured Watson x LLM instance ready for text generation
                   with the specified model ID, credentials, and parameters.
    Note:
        This function requires that model_id, credentials, and credential_params
        are already defined in the outer scope.
    """
    # Set the necessary parameters for the model
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # Specify the maximum number of tokens to generate
        GenParams.TEMPERATURE: 0.5,  # Set the randomness of the model's responses
    }

    # Wrap the model into WatsonxLLM inference
    model = ModelInference(
        model_id=model_id,  # Use the specified model ID
        credentials=credentials,  # Use the provided credentials for authentication
        params=parameters,  # Pass the parameters for model configuration
        project_id=credential_params['project_id']  # Specify the project ID for context
    )

    return WatsonxLLM(watsonx_model=model)  # Return the wrapped model for use

def document_loader(files):
    """
    Load documents from PDF files.
    Args:
        files (str or list): Path to a PDF file or a list of paths to PDF files.
            If a single file path is provided, it will be converted to a list.
    Returns:
        list: A list of loaded documents from all the provided PDF files.
    Example:
        >>> docs = document_loader("path/to/file.pdf")
        >>> docs = document_loader(["path/to/file1.pdf", "path/to/file2.pdf"])
    """
    if not isinstance(files, list):
        files = [files]  # Ensure it's always a list to handle single or multiple file inputs
    
    all_docs = []  # Initialize an empty list to store all loaded documents
    for file in files:  # Iterate through each file in the list
        loader = PyPDFLoader(file)  # Create a loader instance for the current PDF file
        docs = loader.load()  # Load the documents from the PDF file
        all_docs.extend(docs)  # Add the loaded documents to the all_docs list
    return all_docs  # Return the list of all loaded documents

def text_splitter(documents):
    """
    Splits documents into smaller, overlapping chunks for processing.
    
    This function uses RecursiveCharacterTextSplitter to divide input documents
    into manageable chunks with specified size and overlap.
    
    Args:
        documents (list): A list of Document objects to be split into chunks
    
    Returns:
        list: A list of Document chunks with the content split according to 
              the specified chunk size and overlap
    
    Example:
        >>> docs = [Document(page_content="...")]
        >>> chunks = text_splitter(docs)
    """
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Set the size of each text chunk
        chunk_overlap=50,  # Set the overlap between chunks
        length_function=len,  # Use the length function to determine chunk sizes
    )
    chunks = text_splitter.split_documents(documents)  # Split the documents into chunks
    return chunks  # Return the list of text chunks

def get_embeddings():
    """
    Creates and configures a Watson x Embeddings model for text embedding generation.
    
    The function sets up parameters for embedding generation, including truncation options
    and return options. It initializes a WatsonxEmbeddings instance with the slate-125m-english-rtrvr
    model and authentication credentials from credential_params.
    
    Returns:
        WatsonxEmbeddings: Configured embedding model ready to generate text embeddings.
        
    Note:
        This function requires credential_params to be defined in the scope with:
        - url: The Watson service URL
        - api_key: API key for authentication
        - project_id: The Watson project ID
        - username: The username for authentication
    """
    # Set parameters for embedding generation
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 1,  # Specify truncation of input tokens
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}  # Set return options for input text
    }
    # Initialize WatsonxEmbeddings with the specified parameters
    watsonxembedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",  # Specify the model ID for embeddings
        url=credential_params['url'],  # Use the provided URL for the service
        apikey=credential_params['api_key'],  # Use the API key for authentication
        project_id=credential_params['project_id'],  # Specify the project ID
        username=credential_params['username'],  # Use the provided username
        params=embed_params,  # Pass the embedding parameters
    )
    return watsonxembedding  # Return the embedding model

def create_vector_store(chunks):
    """
    Creates a vector store from document chunks using embeddings.
    
    This function takes a list of document chunks, retrieves an embedding model,
    and creates a persistent Chroma vector database in the "db" directory.
    
    Args:
        chunks (list): A list of document chunks to be stored in the vector database.
        
    Returns:
        Chroma: A Chroma vector store containing the embedded document chunks.
    """
    # Get the embedding model
    embedding_model = get_embeddings()  # Retrieve the embedding model
    # Create a Chroma vector store from document chunks using our embedding model
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="db")
    # Save the vector store to disk for persistence between sessions
    vectordb.persist()
    return vectordb  # Return the vector store

def get_retrieval(files):
    """
    Creates a retriever object from a list of files.
    This function processes the input files by loading their content, splitting them into
    manageable chunks, creating a vector store, and finally setting up a retriever.
    Args:
        files (list): A list of file paths to process, typically PDF documents.
    Returns:
        Retriever: A retriever object that can be used to perform semantic searches
                   on the content of the provided files.
    Note:
        This function relies on document_loader, text_splitter, and create_vector_store
        functions which should be defined elsewhere in the code.
    """

    documents = document_loader(files)      # Load all PDFs using the document_loader function
    chunks = text_splitter(documents)       # Split the loaded documents into manageable chunks
    vectordb = create_vector_store(chunks)  # Create a vector store from the chunks for efficient retrieval
    retriever = vectordb.as_retriever()     # Convert the vector store into a retriever object
    return retriever  # Return the retriever for further use


def get_memory(memory_key="chat_history", return_messages=True, output_key="answer"):
    """
    Creates and initializes a conversation memory buffer for chat interactions.
    
    This function creates a ConversationBufferMemory object that stores the chat
    history and allows retrieval of previous interactions in the conversation.
    
    Parameters:
    -----------
    memory_key : str, optional (default="chat_history")
        The key under which the conversation history will be stored.
    
    return_messages : bool, optional (default=True)
        Whether to return the history as message objects or as a string.
        If True, returns a list of message objects.
        If False, returns a string representation.
    
    output_key : str, optional (default="answer")
        The key used to store the model's response in the memory.
    
    Returns:
    --------
    ConversationBufferMemory
        An initialized memory object configured with the specified parameters.
    """
    # Initialize conversation memory with specified parameters
    memory = ConversationBufferMemory(
        memory_key=memory_key,  # Set the key for memory storage
        return_messages=return_messages,  # Specify whether to return messages
        output_key=output_key  # Set the output key for responses
    )
    return memory  # Return the memory object

def retriever_qa(file, query, history=None):
    """
    Performs a question-answering task using a conversational retrieval chain.
    This function creates a conversational retrieval chain that processes a query
    against a document provided by the file parameter. It uses the document retriever
    to find relevant information and generates an answer based on the query and
    conversation history.
    Args:
        file: The path to the document file to query against.
        query (str): The question or query to be answered.
        history (list, optional): The conversation history as a list of (query, response)
                                  tuples. Defaults to an empty list.
    Returns:
        tuple: A tuple containing:
            - answer (str): The generated answer to the query.
            - history (list): The updated conversation history with the new query-answer pair.
            - sources (str): A formatted string containing the source documents used for
                            answering the query, including page numbers and content snippets.
    Note:
        Using mutable default arguments (like history=[]) can lead to unexpected behavior
        if the function is called multiple times without explicitly providing a history argument.
    """
    if history is None:
        history = []  # Initialize history as an empty list if not provided
    llm = get_llm()  # Retrieve the language model
    retriever_obj = get_retrieval(file)  # Get the document retriever
    memory = get_memory(output_key="answer")  # Initialize memory for conversation
    # Create a conversational retrieval chain with the specified components
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Language model
        chain_type="stuff",  # Specify the type of chain
        retriever=retriever_obj,  # Retriever for document access
        memory=memory,  # Memory for conversation context
        return_source_documents=True  # Return source documents used for answering
    )
    # Invoke the QA chain with the query and conversation history
    response = qa.invoke({"question": query, "chat_history": history})

    # Update the conversation history with the new query-answer pair
    history.append((query, response["answer"]))
    # Format the source documents for output (showing page and snippet)
    sources = "\n\n".join(
        [f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}..." for doc in response["source_documents"]]
    )
    # Return the answer, updated history, and formatted sources
    return response["answer"], history, sources

with gr.Blocks(css=".gradio-container {font-family: 'Segoe UI', sans-serif;}") as rag_app:  # Create a block layout for the Gradio app
    gr.Markdown("## 📘 Document Chatbot with Memory\nUpload a PDF and ask questions interactively. The bot will remember previous queries and show sources.")  # Markdown header and description

    with gr.Row():  # Create a row layout for input components
        with gr.Column(scale=2):  # Create a column for input components
            file_input = gr.File(  # File input for uploading PDFs
                label="📂 Upload PDF(s)", 
                file_types=[".pdf"], 
                file_count="multiple", 
                type="filepath"
            )
            
            chatbot = gr.Chatbot(  # Chatbot component for displaying conversation history
                label="💬 Chat History", 
                height=400
            )
            
            query = gr.Textbox(  # Textbox for user input queries
                label="Ask a Question", 
                placeholder="Type your question here...", 
                lines=2
            )
            # Create a row to hold the buttons side by side
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit")  # Button for submitting the query
                clear_btn = gr.Button("🗑️ Clear Chat")  # Clear button next to submit button
            
        with gr.Column(scale=1):  # Create a column for output components
            response_box = gr.Textbox(  # Textbox for displaying model responses
                label="🤖 Answer", 
                lines=6, 
                placeholder="The model's response will appear here..."
            )
            
            sources_box = gr.Textbox(  # Textbox for displaying source document snippets
                label="📑 Sources", 
                lines=8, 
                placeholder="Relevant document snippets will appear here..."
            )

    def chat_fn(file, query, history=None):  # Function to handle chat interactions
        """
        Handle chat interactions by retrieving answers to user queries.

        This function processes a user query against a specific file, maintains conversation history, 
        and returns relevant information along with sources.

        Args:
            file: The file or document to search for answers.
            query (str): The user's question or query.
            history (list, optional): Previous conversation history. Defaults to an empty list.

        Returns:
            tuple: A tuple containing:
                - answer (str): The response to the user's query.
                - history (list): Updated conversation history.
                - sources (list): References or sources used to generate the answer.
        """
        if history is None:
            history = []  # Initialize history as an empty list if not provided
        answer, history, sources = retriever_qa(file, query, history)  # Call the QA function
        return answer, history, sources  # Return the answer, updated history, and sources
        
    submit_btn.click(  # Set up the button click event
        chat_fn,  # Function to call on click
        inputs=[file_input, query, chatbot],  # Inputs for the function
        outputs=[response_box, chatbot, sources_box],  # Outputs for the function
        show_progress='full'  # Show progress during processing
    )
    
    clear_btn.click(lambda: ([], None, ""), outputs=[chatbot, response_box, sources_box])


rag_app.launch(server_name="127.0.0.1", server_port=7860)  # Launch the Gradio app on the specified server and port
