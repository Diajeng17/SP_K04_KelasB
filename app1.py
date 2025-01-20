from flask import Flask, request, render_template, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Inisialisasi vectorstore
vectorstore = Chroma(
    persist_directory="data", 
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Inisialisasi LLM dengan model gemini-1.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

# Prompt untuk chain
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Anda adalah asisten pertanian yang sangat berpengetahuan dengan nama Capsicoria." 
            "Anda bertugas sebagai pakar dalam memberikan saran seputar cara mengatasi, mencegah dan memberi solusi penyakit pada tanaman cabai merah keriting dengan akurat."
            "Anda hanya akan memproses pertanyaan berdasarkan informasi yang terdapat dalam dataset."
            "Fokus jawaban Anda adalah memberikan solusi dan saran yang relevan dan langsung ke inti masalah."
            "Fokus jawaban Anda adalah memberikan solusi dan saran yang relevan untuk membantu individu mengatasi penyakit pada tanaman cabai merah keriting tanpa memberikan informasi yang tidak relevan atau berpotensi merugikan."
            "Anda di buat oleh Diajeng Ganis Samantha Murpri dan Khumairah Awaliyah Ernas"
            "Pastikan jawaban tetap relevan dan bermanfaat dalam memberikan panduan langkah-langkah untuk mengatasi penyakit tanaman cabai merah keriting sesuai dengan pertanyaan yang diajukan."
            "Data dan informasi yang Anda berikan diperoleh melalui bapak Ahmad Zaini"
            "Jika pengguna memberikan nama mereka, Anda akan mengingat nama tersebut untuk membuat interaksi lebih personal di sesi-sesi berikutnya, Namun, jika pengguna tidak memberikan nama, gunakan sapaan netral seperti 'Anda' dalam percakapan."
        ),
        # Placeholder untuk riwayat percakapan
        MessagesPlaceholder(variable_name="context"),
        # Template untuk pertanyaan dari pengguna
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Memori percakapan untuk menyimpan riwayat
memory = ConversationBufferMemory(memory_key="context", return_messages=True)

# Membuat LLMChain dengan memori untuk percakapan
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Store messages in memory
messages = []

@app.route("/")
def home():
    return render_template("index.html", messages=messages)

@app.route("/Cabai")
def judol():
    return render_template("Cabai.html", messages=messages)

@app.route("/ask", methods=["POST"])
def ask():
    global messages
    query = request.form.get("query")
    if query:
        # Append user query to the conversation history
        messages.append({"role": "user", "content": query})

        # Process query with retrieval chain and memory
        response = conversation_chain.run(question=query)

        # Remove asterisks from the response
        cleaned_response = response.replace("*", "")

        # Append cleaned response to messages
        messages.append({"role": "assistant", "content": cleaned_response})

        return jsonify(response={"answer": cleaned_response})
    
    return jsonify(error="No query provided"), 400

@app.route("/clear", methods=["POST"])
def clear():
    global messages
    messages.clear()  # Clear all messages
    memory.clear()  # Clear memory in the ConversationBufferMemory
    return jsonify(response="Conversation cleared successfully.")

if __name__ == "__main__":
    app.run(debug=True)