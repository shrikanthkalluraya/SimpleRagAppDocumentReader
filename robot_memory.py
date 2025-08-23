import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class BookRobot:
    def __init__(self):
        # Give our robot a brain and memory!
        self.brain = ChatOpenAI(temperature=0)  # The thinking part
        self.memory_maker = OpenAIEmbeddings()  # Turns words into robot memories

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
            self.memory_maker
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


if __name__ == "__main__":
    robot = BookRobot()
    print("Book robot is ready! 🤖")

