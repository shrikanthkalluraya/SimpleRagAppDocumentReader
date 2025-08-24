import streamlit as st
from typing import Dict, Any, List, Literal
from typing_extensions import TypedDict
from transformers import pipeline
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sys

# Install LangGraph: pip install langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class BookRobotState(TypedDict):
    """The shared state that flows through our LangGraph"""
    # Input
    question: str
    book_content: str
    
    # Processing data
    book_chunks: List[str]
    relevant_info: str
    question_type: str
    analysis: str
    
    # Robot responses
    librarian_response: str
    detective_response: str
    writer_response: str
    wisdom_response: str
    
    # Final output
    final_answer: str
    
    # Control flow
    next_action: str
    step_count: int


class BookRobotTeam:
    """🏰 The Robot Kingdom using REAL LangGraph!"""
    
    def __init__(self):
        # Initialize embeddings for the librarian
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.memory_box = None
        
        # Build the LangGraph workflow
        self.workflow = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the actual LangGraph StateGraph"""
        
        # Create the graph
        workflow = StateGraph(BookRobotState)
        
        # Add all our robot nodes
        workflow.add_node("librarian", self.librarian_node)
        workflow.add_node("detective", self.detective_node)  
        workflow.add_node("writer", self.writer_node)
        workflow.add_node("wisdom", self.wisdom_node)
        workflow.add_node("king", self.king_node)
        
        # Define the workflow edges (who talks to whom)
        workflow.add_edge(START, "librarian")  # Always start with librarian
        workflow.add_edge("librarian", "detective")  # Librarian → Detective
        
        # Detective makes conditional decisions
        workflow.add_conditional_edges(
            "detective",
            self._route_after_detective,  # Function that decides next step
            {
                "writer": "writer",      # If simple question → Writer
                "wisdom": "wisdom",      # If complex question → Wisdom Robot
            }
        )
        
        workflow.add_edge("writer", "king")    # Writer → King
        workflow.add_edge("wisdom", "king")    # Wisdom → King  
        workflow.add_edge("king", END)         # King finishes the workflow
        
        # Add memory (so robots remember previous conversations)
        memory = MemorySaver()
        
        # Compile the graph
        return workflow.compile(checkpointer=memory)
    
    def _route_after_detective(self, state: BookRobotState) -> Literal["writer", "wisdom"]:
        """🕵️ Detective decides which robot should work next"""
        question_lower = state["question"].lower()
        
        # If it's a complex reasoning question, send to wisdom robot
        if any(word in question_lower for word in ["why", "analyze", "meaning", "significance", "interpret"]):
            return "wisdom"
        else:
            # Simple questions go to writer
            return "writer"
    
    # ===================
    # 🤖 ROBOT NODES 
    # ===================
    
    def librarian_node(self, state: BookRobotState) -> BookRobotState:
        """📚 Librarian Robot: Reads books and finds relevant information"""
        print("📚 LIBRARIAN ROBOT: Processing book and searching for information...")

        relevant_chunks = []  # Always defined
        # Process book if needed
        if state["book_content"] and not self.memory_box:
            print("📖 Reading and memorizing the book...")
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(state["book_content"])
            
            self.memory_box = Chroma.from_texts(
                chunks, 
                self.embeddings,
                persist_directory="./langgraph_robot_memory"
            )
            
            state["book_chunks"] = chunks
            print(f"✅ Memorized {len(chunks)} book sections!")
        
        # Find relevant information for the question
        relevant_info = ""
        if state["question"] and self.memory_box:
            print(f"🔍 Searching for: '{state['question']}'")
            relevant_chunks = self.memory_box.similarity_search(state["question"], k=3)
            relevant_info = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            print(f"✅ Found {len(relevant_chunks)} relevant sections")
        
        # Update state
        state["relevant_info"] = relevant_info
        state["librarian_response"] = f"📚 Found {len(relevant_chunks) if self.memory_box else 0} relevant book sections about your question."
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    def detective_node(self, state: BookRobotState) -> BookRobotState:
        """🕵️ Detective Robot: Analyzes the question and evidence"""
        print("🕵️ DETECTIVE ROBOT: Analyzing question and evidence...")
        
        question_lower = state["question"].lower()
        relevant_info = state["relevant_info"]
        
        # Analyze question type
        if any(word in question_lower for word in ["who", "character", "person"]):
            question_type = "CHARACTER_IDENTIFICATION"
            analysis = "This question asks about people or characters in the story."
            
        elif any(word in question_lower for word in ["what", "describe", "explain"]):
            question_type = "DESCRIPTION_REQUEST"  
            analysis = "This question wants a description or explanation of something."
            
        elif any(word in question_lower for word in ["where", "place", "location", "setting"]):
            question_type = "LOCATION_INQUIRY"
            analysis = "This question is about places or settings in the story."
            
        elif any(word in question_lower for word in ["why", "reason", "because", "purpose"]):
            question_type = "REASONING_QUESTION"
            analysis = "This is a complex question that needs deep reasoning and analysis."
            
        elif any(word in question_lower for word in ["how", "method", "way", "process"]):
            question_type = "PROCESS_QUESTION" 
            analysis = "This question asks about how something happens or works."
            
        elif any(word in question_lower for word in ["summarize", "summary", "overview"]):
            question_type = "SUMMARY_REQUEST"
            analysis = "This question wants a summary or overview."
            
        else:
            question_type = "GENERAL_QUESTION"
            analysis = "This is a general question about the book content."
        
        print(f"🔍 Analysis: {question_type} - {analysis}")
        
        # Update state
        state["question_type"] = question_type
        state["analysis"] = analysis
        state["detective_response"] = f"🕵️ Analysis complete: {question_type}. {analysis}"
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    def writer_node(self, state: BookRobotState) -> BookRobotState:
        """✍️ Writer Robot: Creates well-written responses"""
        print("✍️ WRITER ROBOT: Crafting a beautiful response...")
        
        question = state["question"]
        question_type = state["question_type"]
        relevant_info = state["relevant_info"]
        
        # Write response based on question type
        if question_type == "CHARACTER_IDENTIFICATION":
            response = self._write_character_response(relevant_info)
        elif question_type == "DESCRIPTION_REQUEST":
            response = self._write_description_response(question, relevant_info)
        elif question_type == "LOCATION_INQUIRY":
            response = self._write_location_response(relevant_info)
        elif question_type == "SUMMARY_REQUEST":
            response = self._write_summary_response(relevant_info)
        else:
            response = self._write_general_response(question, relevant_info)
            
        print("✅ Response crafted with care!")
        
        # Update state
        state["writer_response"] = response
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    def wisdom_node(self, state: BookRobotState) -> BookRobotState:
        """🧠 Wisdom Robot: Provides deep analysis and insights"""
        print("🧠 WISDOM ROBOT: Contemplating the deeper meanings...")
        
        question = state["question"]
        relevant_info = state["relevant_info"]
        question_type = state["question_type"]
        
        # Provide thoughtful analysis
        wisdom_response = f"""🧠 **Deep Wisdom Analysis**

**Your Question**: {question}

**Deeper Context**: Looking beyond the surface, this {question_type.lower().replace('_', ' ')} touches on important themes in the narrative.

**Key Insights from the Text**:
{relevant_info[:400]}...

**Philosophical Consideration**: The answer involves understanding both the literal events and their symbolic significance within the broader story context.

**Thoughtful Reflection**: Great literature often contains layers of meaning. Your question invites us to explore not just what happens, but why it matters in the larger tapestry of the story."""
        
        print("✅ Wisdom shared!")
        
        # Update state  
        state["wisdom_response"] = wisdom_response
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    def king_node(self, state: BookRobotState) -> BookRobotState:
        """👑 King Robot: Coordinates everything and provides final answer"""
        print("👑 KING ROBOT: Assembling the royal decree...")
        
        question = state["question"]
        
        # Collect all responses
        team_reports = []
        
        if state.get("librarian_response"):
            team_reports.append(f"📚 **Librarian**: {state['librarian_response']}")
            
        if state.get("detective_response"):
            team_reports.append(f"🕵️ **Detective**: {state['detective_response']}")
            
        if state.get("writer_response"):
            team_reports.append(f"✍️ **Writer**: {state['writer_response']}")
            
        if state.get("wisdom_response"):
            team_reports.append(f"🧠 **Wisdom**: {state['wisdom_response']}")
        
        # Create comprehensive final answer
        final_answer = f"""👑 **ROYAL TEAM RESPONSE**

**Your Question**: {question}

**Team Analysis Summary**:
{chr(10).join(team_reports)}

**Final Royal Answer**: Based on my expert team's collaborative analysis, here is the comprehensive response to your inquiry about the book."""
        
        print("✅ Royal decree complete!")
        
        # Update state
        state["final_answer"] = final_answer
        state["step_count"] = state.get("step_count", 0) + 1
        
        return state
    
    # ===================
    # 💬 HELPER METHODS
    # ===================
    
    def _write_character_response(self, text: str) -> str:
        words = text.split()
        potential_names = [word.strip('.,!?') for word in words if word.istitle() and len(word) > 2 and word.isalpha()]
        
        if potential_names:
            main_character = potential_names[0]
            return f"✍️ **Character Analysis**: Based on the book, the main character appears to be **{main_character}**. The narrative centers around their experiences and development throughout the story."
        return "✍️ **Character Analysis**: The text discusses various characters, with their roles and relationships being central to the narrative."
    
    def _write_description_response(self, question: str, text: str) -> str:
        return f"✍️ **Detailed Description**: {text[:300]}...\n\nThis passage provides rich context and imagery that directly relates to your question."
    
    def _write_location_response(self, text: str) -> str:
        return f"✍️ **Setting Analysis**: The story takes place in the world described in the text:\n\n{text[:250]}...\n\nThe setting plays a crucial role in shaping the narrative."
        
    def _write_summary_response(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()][:3]
        summary = '. '.join(sentences) + '.'
        return f"✍️ **Summary**: {summary}"
        
    def _write_general_response(self, question: str, text: str) -> str:
        return f"✍️ **Response**: Based on the book content, here's what's most relevant to your question:\n\n{text[:300]}..."
    
    # ===================
    # 🚀 PUBLIC METHODS
    # ===================
    
    def process_book(self, book_content: str) -> str:
        """Give the robot team a book to read"""
        # Initialize state for book processing
        initial_state = BookRobotState(
            question="",  # No question yet, just processing book
            book_content=book_content,
            book_chunks=[],
            relevant_info="",
            question_type="",
            analysis="",
            librarian_response="",
            detective_response="",
            writer_response="", 
            wisdom_response="",
            final_answer="",
            next_action="",
            step_count=0
        )
        
        # Just let librarian process the book
        updated_state = self.librarian_node(initial_state)
        
        return f"🏰 Robot Team has read your book! Memorized {len(updated_state.get('book_chunks', []))} sections."
    
    def ask_question(self, question: str) -> str:
        """Ask the robot team a question using LangGraph workflow"""
        print(f"\n🏰 LANGGRAPH WORKFLOW STARTING...")
        print(f"❓ Question: {question}")
        
        # Initial state
        config = {"configurable": {"thread_id": "book_robot_session"}}
        
        initial_state = BookRobotState(
            question=question,
            book_content="",  # Book already processed
            book_chunks=[],
            relevant_info="",
            question_type="",
            analysis="",
            librarian_response="",
            detective_response="",
            writer_response="",
            wisdom_response="",
            final_answer="",
            next_action="",
            step_count=0
        )
        
        # Run the LangGraph workflow
        print("\n🔄 Executing LangGraph workflow...")
        result = self.workflow.invoke(initial_state, config=config)
        
        print(f"✅ Workflow complete! Final result ready.")
        return result["final_answer"]


# Streamlit Interface
def main():
    st.title("🏰 REAL LangGraph BookRobot Kingdom! ⚡")
    st.write("**Powered by LangGraph's StateGraph with START → END workflow!**")
    
    # Show workflow diagram
    with st.expander("🕸️ View LangGraph Workflow", expanded=False):
        st.code("""
🚀 LangGraph Workflow:

START → 📚 Librarian → 🕵️ Detective → [Decision Point]
                                          ↓
                                    ✍️ Writer OR 🧠 Wisdom
                                          ↓
                                      👑 King → END
        """)
    
    # Initialize robot team
    if 'robot_team' not in st.session_state:
        with st.spinner("🏰 Initializing LangGraph Robot Team..."):
            st.session_state.robot_team = BookRobotTeam()
    
    # File upload
    uploaded_file = st.file_uploader("📚 Upload Book for Robot Team", type="txt")
    
    if uploaded_file is not None:
        book_content = str(uploaded_file.read(), "utf-8")
        
        if st.button("📖 Process Book with LangGraph"):
            with st.spinner("🤖 Robot team processing book..."):
                result = st.session_state.robot_team.process_book(book_content)
            st.success(result)
            st.session_state.book_processed = True
    
    # Question interface
    if st.session_state.get('book_processed'):
        st.subheader("🤖 Ask the LangGraph Robot Team")
        
        question = st.text_input("What would you like to know?")
        
        if st.button("⚡ Execute LangGraph Workflow") and question:
            with st.spinner("🔄 LangGraph workflow executing..."):
                answer = st.session_state.robot_team.ask_question(question)
            
            st.markdown("---")
            st.write("🏆 **LANGGRAPH FINAL ANSWER:**")
            st.markdown(answer)
    
    # Sidebar
    st.sidebar.title("⚡ Real LangGraph Features")
    st.sidebar.write("""
    **This uses ACTUAL LangGraph:**
    
    ✅ **StateGraph** - Real graph structure
    ✅ **START/END** - Proper workflow boundaries  
    ✅ **Conditional Edges** - Smart routing
    ✅ **Memory** - Remembers conversations
    ✅ **State Management** - Shared data flow
    
    **Workflow Steps:**
    1. START → Librarian (always first)
    2. Librarian → Detective (finds info)
    3. Detective → Writer OR Wisdom (decides)
    4. Writer/Wisdom → King (final assembly)
    5. King → END (workflow complete)
    
    **Key LangGraph Concepts:**
    - 🏷️ **TypedDict State** 
    - 🔄 **Conditional Routing**
    - 💾 **MemorySaver Checkpoints**
    - 🕸️ **Graph Compilation**
    """)


# Demo
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("⚡" + "="*60 + "⚡")
        print("         REAL LANGGRAPH ROBOT KINGDOM DEMO")  
        print("⚡" + "="*60 + "⚡")
        
        team = BookRobotTeam()
        
        # Sample book  
        sample_book = """
        Sherlock Holmes, the famous detective of Baker Street, sat in his armchair 
        examining a peculiar case. Dr. Watson entered the room carrying the morning newspaper.
        
        "Holmes," said Watson, "there's been another mysterious theft in London. 
        The victim claims nothing was stolen, yet they feel something important is missing."
        
        Holmes raised an eyebrow. "Fascinating, Watson. A theft where nothing is taken, 
        yet something is lost. This requires our immediate attention."
        
        The detective stood up, grabbed his coat and deerstalker hat. 
        "Come, Watson. The game is afoot!"
        """
        
        print("📚 Processing Sherlock Holmes story...")
        result = team.process_book(sample_book)
        print(result)
        
        questions = [
            "Who are the main characters?",
            "What is the mysterious case about?",
            "Why is this case interesting to Holmes?"
        ]
        
        for question in questions:
            print(f"\n{'='*70}")
            print(f"❓ QUESTION: {question}")
            print('='*70)
            
            answer = team.ask_question(question)
            print(f"\n{answer}")
            
    else:
        main()
