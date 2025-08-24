import streamlit as st
from typing import Dict, Any, List
from dataclasses import dataclass
from transformers import pipeline
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
import sys


@dataclass
class RobotMessage:
    """Messages that robots send to each other"""
    from_robot: str
    to_robot: str
    content: str
    message_type: str  # "question", "answer", "task", "result"
    data: Dict[str, Any] = None


class RobotKingdomState:
    """The shared memory/state that all robots can access"""
    
    def __init__(self):
        self.messages: List[RobotMessage] = []
        self.book_content = ""
        self.book_chunks = []
        self.current_question = ""
        self.robot_responses = {}
        self.final_answer = ""
        self.conversation_history = []
        self.next_robot = "librarian"  # Which robot should work next
        
    def add_message(self, message: RobotMessage):
        self.messages.append(message)
        
    def get_messages_for_robot(self, robot_name: str) -> List[RobotMessage]:
        return [msg for msg in self.messages if msg.to_robot == robot_name]
        
    def set_robot_response(self, robot_name: str, response: str):
        self.robot_responses[robot_name] = response


class LibrarianRobot:
    """📚 The Librarian Robot - Reads and remembers books"""
    
    def __init__(self):
        self.name = "librarian"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.memory_box = None
        
    def work(self, state: RobotKingdomState) -> RobotKingdomState:
        """Librarian's job: Read books and find relevant information"""
        print(f"📚 {self.name.upper()} ROBOT: Starting my work...")
        
        # If we have a book but no memory box, process the book
        if state.book_content and not self.memory_box:
            print("📖 Reading and memorizing the book...")
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(state.book_content)
            
            self.memory_box = Chroma.from_texts(
                chunks, 
                self.embeddings,
                persist_directory="./robot_kingdom_memory"
            )
            state.book_chunks = chunks
            print(f"✅ Memorized {len(chunks)} book sections!")
            
        # If we have a question, find relevant information
        if state.current_question and self.memory_box:
            print(f"🔍 Searching for information about: {state.current_question}")
            relevant_chunks = self.memory_box.similarity_search(state.current_question, k=3)
            relevant_text = "\n".join([chunk.page_content for chunk in relevant_chunks])
            
            response = f"I found {len(relevant_chunks)} relevant sections about your question. Here's what I discovered:\n\n{relevant_text}"
            state.set_robot_response(self.name, response)
            
            # Send message to detective robot
            message = RobotMessage(
                from_robot=self.name,
                to_robot="detective", 
                content="I found relevant book information. Please analyze it.",
                message_type="task",
                data={"relevant_text": relevant_text}
            )
            state.add_message(message)
            state.next_robot = "detective"
            
        return state


class DetectiveRobot:
    """🕵️ The Detective Robot - Analyzes information and asks follow-ups"""
    
    def __init__(self):
        self.name = "detective"
        
    def work(self, state: RobotKingdomState) -> RobotKingdomState:
        """Detective's job: Analyze information and determine next steps"""
        print(f"🕵️ {self.name.upper()} ROBOT: Investigating...")
        
        # Get messages from librarian
        my_messages = state.get_messages_for_robot(self.name)
        
        if my_messages:
            latest_message = my_messages[-1]
            relevant_text = latest_message.data.get("relevant_text", "")
            
            # Analyze the question type
            question_lower = state.current_question.lower()
            
            analysis = ""
            next_robot_choice = "writer"
            
            if any(word in question_lower for word in ["who", "character", "person"]):
                analysis = "This is a CHARACTER question. I need to identify people/characters."
                next_robot_choice = "writer"
                
            elif any(word in question_lower for word in ["what", "describe", "explain"]):
                analysis = "This is a DESCRIPTION question. I need to explain something."
                next_robot_choice = "writer"
                
            elif any(word in question_lower for word in ["why", "because", "reason"]):
                analysis = "This is a REASONING question. I should get the wisdom robot involved."
                next_robot_choice = "wisdom"
                
            elif any(word in question_lower for word in ["summarize", "summary", "overview"]):
                analysis = "This is a SUMMARY question. Perfect job for the writer robot."
                next_robot_choice = "writer"
            
            response = f"🕵️ DETECTIVE ANALYSIS:\n{analysis}\n\nRelevant information found: {len(relevant_text)} characters of text."
            state.set_robot_response(self.name, response)
            
            # Send to appropriate next robot
            message = RobotMessage(
                from_robot=self.name,
                to_robot=next_robot_choice,
                content=f"Please handle this {question_lower.split()[0].upper()} question.",
                message_type="task", 
                data={"analysis": analysis, "relevant_text": relevant_text}
            )
            state.add_message(message)
            state.next_robot = next_robot_choice
            
        return state


class WriterRobot:
    """✍️ The Writer Robot - Creates clear, well-written responses"""
    
    def __init__(self):
        self.name = "writer"
        
    def work(self, state: RobotKingdomState) -> RobotKingdomState:
        """Writer's job: Create well-written responses"""
        print(f"✍️ {self.name.upper()} ROBOT: Crafting response...")
        
        my_messages = state.get_messages_for_robot(self.name)
        
        if my_messages:
            latest_message = my_messages[-1]
            relevant_text = latest_message.data.get("relevant_text", "")
            analysis = latest_message.data.get("analysis", "")
            
            # Create a well-structured response
            if "CHARACTER" in analysis:
                response = self._write_character_response(state.current_question, relevant_text)
            elif "DESCRIPTION" in analysis:
                response = self._write_description_response(state.current_question, relevant_text)
            elif "SUMMARY" in analysis:
                response = self._write_summary_response(relevant_text)
            else:
                response = self._write_general_response(state.current_question, relevant_text)
                
            state.set_robot_response(self.name, response)
            
            # Send to king robot for final approval
            message = RobotMessage(
                from_robot=self.name,
                to_robot="king",
                content="I've written a response. Please review and finalize.",
                message_type="result",
                data={"written_response": response}
            )
            state.add_message(message)
            state.next_robot = "king"
            
        return state
        
    def _write_character_response(self, question: str, text: str) -> str:
        # Extract potential character names (simple approach)
        words = text.split()
        potential_names = [word for word in words if word.istitle() and len(word) > 2 and word.isalpha()]
        
        if potential_names:
            main_character = potential_names[0]
            return f"📖 Based on the book, the main character appears to be **{main_character}**. The text mentions them in the context of the story events."
        return f"📖 The text discusses various characters, though the specific main character isn't immediately clear from this section."
        
    def _write_description_response(self, question: str, text: str) -> str:
        return f"📖 Based on the book content:\n\n{text[:300]}...\n\nThis provides context for your question about the story elements."
        
    def _write_summary_response(self, text: str) -> str:
        sentences = text.split('.')[:3]  # First 3 sentences
        summary = '. '.join(sentences) + '.'
        return f"📖 **Summary**: {summary}"
        
    def _write_general_response(self, question: str, text: str) -> str:
        return f"📖 **Answer**: Based on the book content, here's what I found relevant to your question:\n\n{text[:200]}..."


class WisdomRobot:
    """🧠 The Wisdom Robot - Provides thoughtful, deeper analysis"""
    
    def __init__(self):
        self.name = "wisdom"
        
    def work(self, state: RobotKingdomState) -> RobotKingdomState:
        """Wisdom's job: Provide thoughtful analysis and insights"""
        print(f"🧠 {self.name.upper()} ROBOT: Contemplating deeply...")
        
        my_messages = state.get_messages_for_robot(self.name)
        
        if my_messages:
            latest_message = my_messages[-1]
            relevant_text = latest_message.data.get("relevant_text", "")
            
            # Provide deeper analysis
            wisdom_response = f"""🧠 **Deep Analysis**: 

Your question touches on important themes in the text. Looking at the broader context:

{relevant_text[:250]}...

**Key Insights:**
- The narrative structure suggests underlying meanings
- Character motivations may be more complex than they first appear  
- The author's choices in language reveal deeper themes

**Thoughtful Response**: The answer to your question likely involves understanding both the literal events and their symbolic meaning within the story's context."""

            state.set_robot_response(self.name, wisdom_response)
            
            # Send to king robot
            message = RobotMessage(
                from_robot=self.name,
                to_robot="king",
                content="I've provided deep analysis. Ready for final decision.",
                message_type="result", 
                data={"wisdom_response": wisdom_response}
            )
            state.add_message(message)
            state.next_robot = "king"
            
        return state


class KingRobot:
    """👑 The King Robot - Coordinates everything and makes final decisions"""
    
    def __init__(self):
        self.name = "king"
        
    def work(self, state: RobotKingdomState) -> RobotKingdomState:
        """King's job: Coordinate the team and provide final answer"""
        print(f"👑 {self.name.upper()} ROBOT: Making royal decision...")
        
        # Collect all robot responses
        all_responses = []
        
        for robot_name, response in state.robot_responses.items():
            all_responses.append(f"**{robot_name.title()} Robot Report:**\n{response}\n")
            
        # Create final comprehensive answer
        final_answer = f"""👑 **ROYAL ROBOT TEAM RESPONSE**

**Question**: {state.current_question}

**Team Analysis:**
{chr(10).join(all_responses)}

**Final Royal Decree**: Based on my team's analysis, here's the comprehensive answer to your question."""
        
        state.final_answer = final_answer
        state.next_robot = "complete"  # Workflow is complete
        
        return state


class RobotKingdom:
    """🏰 The main system that coordinates all robots"""
    
    def __init__(self):
        self.robots = {
            "librarian": LibrarianRobot(),
            "detective": DetectiveRobot(), 
            "writer": WriterRobot(),
            "wisdom": WisdomRobot(),
            "king": KingRobot()
        }
        self.state = RobotKingdomState()
        
    def read_book(self, book_content: str) -> str:
        """Give the kingdom a book to read"""
        self.state.book_content = book_content
        self.state.next_robot = "librarian"
        
        # Let librarian process the book
        self.state = self.robots["librarian"].work(self.state)
        
        return f"🏰 The Robot Kingdom has read your book! {len(self.state.book_chunks)} sections memorized."
        
    def ask_question(self, question: str) -> str:
        """Ask the robot kingdom a question"""
        print(f"\n🏰 ROBOT KINGDOM WORKFLOW STARTING...")
        print(f"❓ Question: {question}")
        
        # Reset for new question
        self.state.current_question = question
        self.state.robot_responses = {}
        self.state.messages = []
        self.state.next_robot = "librarian"
        
        # Run the robot workflow
        max_steps = 10  # Prevent infinite loops
        step = 0
        
        while self.state.next_robot != "complete" and step < max_steps:
            step += 1
            current_robot_name = self.state.next_robot
            
            if current_robot_name in self.robots:
                print(f"\n--- STEP {step}: {current_robot_name.upper()} ROBOT WORKING ---")
                self.state = self.robots[current_robot_name].work(self.state)
            else:
                break
                
        print(f"\n🏰 WORKFLOW COMPLETE after {step} steps!")
        return self.state.final_answer


# Streamlit Interface
def main():
    st.title("🏰 Robot Kingdom BookBot with LangGraph! 🤖")
    st.write("**A TEAM of AI robots working together!**")
    
    # Initialize kingdom
    if 'kingdom' not in st.session_state:
        with st.spinner("🏰 Building Robot Kingdom..."):
            st.session_state.kingdom = RobotKingdom()
    
    # File upload
    uploaded_file = st.file_uploader("📚 Give the Kingdom a Book", type="txt")
    
    if uploaded_file is not None:
        book_content = str(uploaded_file.read(), "utf-8")
        
        if st.button("📖 Royal Reading Command"):
            with st.spinner("👑 Kingdom is reading..."):
                result = st.session_state.kingdom.read_book(book_content)
            st.success(result)
            st.session_state.book_loaded = True
    
    # Question interface
    if st.session_state.get('book_loaded'):
        st.subheader("🤖 Ask the Robot Kingdom")
        
        question = st.text_input("What royal question do you have?")
        
        if st.button("👑 Summon the Robot Team") and question:
            with st.spinner("🏰 Robot team is working together..."):
                with st.expander("👀 Watch the Robot Team Work", expanded=True):
                    # Create a placeholder for real-time updates
                    status_placeholder = st.empty()
                    
                    # Capture print statements (simplified for demo)
                    answer = st.session_state.kingdom.ask_question(question)
                
            st.markdown("---")
            st.write("🏆 **FINAL TEAM ANSWER:**")
            st.markdown(answer)
    
    # Sidebar explanation
    st.sidebar.title("🏰 Robot Kingdom Team")
    st.sidebar.write("""
    **Meet Your Robot Team:**
    
    👑 **King Robot**
    - Coordinates everyone
    - Makes final decisions
    
    📚 **Librarian Robot** 
    - Reads and memorizes books
    - Finds relevant information
    
    🕵️ **Detective Robot**
    - Analyzes questions
    - Decides who should help
    
    ✍️ **Writer Robot**
    - Creates well-written responses
    - Formats answers nicely
    
    🧠 **Wisdom Robot**
    - Provides deep analysis
    - Offers thoughtful insights
    
    **This is LangGraph in action!**
    Each robot is a "node" in the graph, and they communicate through "edges"!
    """)
    
    with st.sidebar.expander("🤖 LangGraph vs LangChain"):
        st.write("""
        **LangChain** 🔗
        - One robot doing everything
        - Linear: Step 1 → 2 → 3
        - Good for simple tasks
        
        **LangGraph** 🕸️
        - Team of robots working together  
        - Graph: Robots can talk to each other
        - Great for complex workflows
        - Each robot has a specialized job
        - Dynamic decision making
        """)


# Demo
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("🏰" + "="*60 + "🏰")
        print("         ROBOT KINGDOM LANGGRAPH DEMO")  
        print("🏰" + "="*60 + "🏰")
        
        kingdom = RobotKingdom()
        
        # Sample book
        sample_book = """
        In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, 
        filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole 
        with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.

        It had a perfectly round door like a porthole, painted green, with a shiny yellow brass knob 
        in the exact middle. The door opened on to a tube-shaped hall like a tunnel: a very comfortable 
        tunnel without smoke, with panelled walls, and floors tiled and carpeted.
        
        This hobbit was a very well-to-do hobbit, and his name was Bilbo Baggins.
        """
        
        print("📚 Teaching the Robot Kingdom about Hobbits...")
        result = kingdom.read_book(sample_book)
        print(result)
        
        questions = [
            "Who is the main character?",
            "What does a hobbit-hole look like?", 
            "Why is this character important?"
        ]
        
        for question in questions:
            print(f"\n{'='*60}")
            print(f"🤔 ASKING: {question}")
            print('='*60)
            
            answer = kingdom.ask_question(question)
            print(f"\n{answer}")
            
    else:
        main()
