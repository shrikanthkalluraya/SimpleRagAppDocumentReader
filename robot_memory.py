import streamlit as st
import sys
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from LocalLLM import LocalLLM, SimpleLocalLLM
from huggingfacellm import HuggingFaceLLM


class BookRobot:
    def __init__(self, use_local_model=True):
        # Give our robot a brain and memory!
        # self.brain = HuggingFaceLLM(
        #     model_name="google/flan-t5-base",  # Free model
        #     hf_token=hf_token
        # )

        # Use FREE local embeddings instead of OpenAI
        self.memory_maker = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        if use_local_model:
            try:
                self.brain = LocalLLM("google/flan-t5-small")  # Small local model
            except Exception as e:
                print(f"⚠️ Could not load local model: {e}")
                print("🔄 Switching to simple pattern-based responses...")
                self.brain = SimpleLocalLLM()
        else:
            self.brain = SimpleLocalLLM()

    def read_book(self, book_text):
        """This is how our robot reads and remembers a book!"""

        # Step 1: Break the book into small pieces (like cutting pizza into slices)
        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # Each piece has 500 characters
            chunk_overlap=50  # Pieces overlap a little (like puzzle pieces)
        )

        pieces = text_splitter.split_text(book_text)

        # Step 2: Turn each piece into a memory our robot can understand
        self.memory_box = Chroma.from_texts(
            pieces,
            self.memory_maker,
            persist_directory="./book_robot_memory"
        )

        return f"I read your book! I broke it into {len(pieces)} pieces to remember better!"

    def answer_question(self, question):
        """This is how our robot answers questions!"""

        # Step 1: Look through memories to find helpful pieces
        helpful_pieces = self.memory_box.similarity_search(question, k=2)

        # Step 2: Put the pieces together
        context = "\n".join([piece.page_content for piece in helpful_pieces])

        # Step 3: Think of an answer using the helpful pieces
        prompt = f"""
        Hi! I'm your friendly book robot! 🤖

        Here's what I remember from the book:
        {context}

        Now, your question is: {question}

        Let me give you a helpful answer based on what I read:
        """

        answer = self.brain.invoke(prompt)
        return answer.content


# Streamlit Web Interface
def main():
    st.title("📚 BookRobot with Free AI! 🤖")
    st.write("Upload a book and ask questions - completely FREE!")

    # Initialize robot
    if 'robot' not in st.session_state:
        with st.spinner("🤖 Initializing BookRobot..."):
            st.session_state.robot = BookRobot()

    # File upload
    uploaded_file = st.file_uploader("Upload a text file (book)", type="txt")

    if uploaded_file is not None:
        # Read the book
        book_content = str(uploaded_file.read(), "utf-8")

        if st.button("📖 Teach Robot This Book"):
            with st.spinner("🤖 Robot is reading the book..."):
                result = st.session_state.robot.read_book(book_content)
            st.success(result)
            st.session_state.book_loaded = True

    # Question interface
    if 'book_loaded' in st.session_state:
        st.subheader("💬 Ask the Robot Questions")

        question = st.text_input("What would you like to know about the book?")

        if st.button("🤔 Ask Robot") and question:
            with st.spinner("🤖 Robot is thinking..."):
                answer = st.session_state.robot.answer_question(question)

            st.write("🤖 **Robot's Answer:**")
            st.write(answer)

    # Information sidebar
    st.sidebar.title("ℹ️ About")
    st.sidebar.write("""
    **FREE BookRobot Features:**
    - 🆓 Uses free Hugging Face models
    - 🧠 Local embeddings (no API costs)
    - 💾 Saves memories locally
    - 📚 Reads any text file
    - 🤖 Answers questions about books

    **No API keys required!**
    """)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Command line demo
        robot = BookRobot()
        print("📚 BookRobot is ready! 🤖")

        # Sample book content
        sample_book = """
            Once upon a time, there was a brave knight named Sir Lancelot. 
            He lived in a castle with King Arthur and the Knights of the Round Table.
            Sir Lancelot was known for his courage and his skill with a sword.
            He went on many adventures to protect the kingdom from dragons and evil wizards.
            The people loved him because he was always kind and helpful.
            One day, he saved a village from a terrible dragon that was burning their homes.
            """

        print("\n📖 Teaching robot a sample story...")
        result = robot.read_book(sample_book)
        print(result)

        # Ask sample questions
        questions = [
            "Who is the main character?",
            "What was Sir Lancelot known for?",
            "What did he save the village from?"
        ]

        print("\n" + "=" * 50)
        print("🤖 ASKING SAMPLE QUESTIONS")
        print("=" * 50)

        for question in questions:
            print(f"\n❓ Question: {question}")
            answer = robot.answer_question(question)
            print(f"🤖 Answer: {answer}")
            print("-" * 30)
    else:
        # Run Streamlit app
        main()