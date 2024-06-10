import streamlit as st
import datetime
from headlamphealth.regression import PredictAnxiety
from headlamphealth.recommend_cards import RecommendCards
from streamlit_card import card
from headlamphealth.load_anxiety_chain import load_anxiety_chain
from dotenv import load_dotenv
import os
import csv

load_dotenv()


# Some variables that can be used.
EMPATHY_CSV = "/Users/abhinavkashyap/abhi/projects/headlamphealth/headlamphealth/data/user_journal/1.csv"
EMBEDDING_STORE_DIRECTORY = f"{os.environ['STORES_DIR']}/empathy_store/"
SAMPLE_PATIENT_DATA = "/Users/abhinavkashyap/abhi/projects/headlamphealth/headlamphealth/sample_data/patients_data.xlsx"
CARDS_CSV = (
    "/Users/abhinavkashyap/abhi/projects/headlamphealth/headlamphealth/data/cards.csv"
)
LOG_CSV = "/Users/abhinavkashyap/abhi/projects/headlamphealth/headlamphealth/data/user_journal/log.csv"

ANXIETY_THRESHOLD = 0.5


st.set_page_config(layout="wide")

chat = load_anxiety_chain(
    empathy_csv=EMPATHY_CSV, embedding_store_directory=EMBEDDING_STORE_DIRECTORY
)


def reset_session_state():
    st.session_state.messages = []


# Linear regression on what the anxiety can be
anxiety_predictor = PredictAnxiety(SAMPLE_PATIENT_DATA)


# This recommends cards based on what you write in the journal
# These cards can walk you through trustd sources of anxiety
# references and tips to manage it
cards_recommender = RecommendCards(CARDS_CSV)


st.title("Log Your Health")


with st.sidebar:
    st.markdown("""
                **Log your health**
                
                Journal your health today. We will help you 
                keep tab of your mental health. Give you more insights on 
                how to manage them. Come back to learn more about yourself 
                and improve 
                1. **Date** - Enter the date that are you are logging. 
                2. **Sleep** - How well did you sleep?
                3. **Diet** - How healthy did you eat?                
                4. **Social** - Did you meet your friends and family?
                5. **Exercise** - Did you exercise?
                """)


# Input element for the date

with st.form("journal_form"):
    # Input element for the Sleep
    date = st.date_input("Date ", value=datetime.datetime.now())
    st.write("### Sleep")
    sleep = st.slider(
        "How well did you sleep today?", min_value=10, max_value=30, step=10
    )

    st.write("### Diet")
    diet = st.slider(
        "How healthy did you eat today?", min_value=10, max_value=30, step=10
    )

    st.write("### Social Activity")
    social = st.slider(
        "Did you meet your friends and family?", min_value=10, max_value=30, step=10
    )

    st.write("### Exercise")
    exercise = st.slider(
        "Moving around helps anxiety. Did you move around?",
        min_value=10,
        max_value=30,
        step=10,
    )

    st.write("### Journal")
    st.write(
        "We have found that jorunaling helps in your journey. Pen down your thoughts"
    )
    with st.expander("Here are some Journal Prompts"):
        st.markdown("""
                    1. What are you thankful for. It is okay even if it is a small part of your life. 

2. What happened Yesterday? Write down mundane details about your life yesterday. 
                    """)

    # Get the user to journal as well
    journal = st.text_area("Write down your thoughts")

    submitted = st.form_submit_button("Submit")


if submitted:
    pred_anxiety = anxiety_predictor.predict(
        sleep=sleep, diet=diet, social=social, exercise=exercise
    )
    print(f"predicted anxiety {pred_anxiety}")
    if pred_anxiety > ANXIETY_THRESHOLD:
        # Show certain helpful cards

        with st.spinner("Finding the Cards"):
            recommendations = cards_recommender.recommend_cards(journal)
        cols = st.columns(3)

        for idx, recommendation in enumerate(recommendations):
            with cols[idx]:
                card(
                    title=recommendation.metadata["heading"],
                    text=recommendation.page_content,
                )

    with open(LOG_CSV, "a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Date", "Sleep", "Diet", "Social", "Exercise", "Journal"]
        )

        file_is_empty = os.path.getsize(LOG_CSV) == 0
        if file_is_empty:
            writer.writeheader()

        # These are the set of new rows
        new_row = [
            {
                "Date": str(date),
                "Sleep": sleep,
                "Diet": diet,
                "Social": social,
                "Exercise": exercise,
                "Journal": journal,
            }
        ]
        writer.writerows(new_row)


st.divider()
st.markdown("## Chat with our Agent")


# This displays the chat history after refreshing
# The history is stored in session_state object
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        with st.expander(label=f"{message['role'].capitalize()} says:", expanded=True):
            st.markdown(message["content"])

if prompt := st.chat_input("What do you want to know about me?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        with st.expander(label="User says: ", expanded=True):
            st.markdown(prompt)


# This steps through the chain stream
# and yields those chunks that have an answer with them
def chunk_generator(stream):
    for chunk in stream:
        if chunk is None:
            break
        if chunk.get("answer"):
            yield chunk["answer"]


if prompt:
    with st.chat_message("assistant"):
        # the answer here is a stream
        with st.spinner("Our Agent is working hard to find an answer"):
            answer = chat.chat_stream(prompt, session_id="abc")
            with st.expander("Assistant Says: ", expanded=True):
                # The last message is the output of the write_stream function
                final_answer = st.write_stream(chunk_generator(answer))
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )
