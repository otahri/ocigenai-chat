from streamlit import session_state as ss
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time
import configparser

st.set_page_config(layout='wide')

config = configparser.ConfigParser()
config.read('/config/app.config')
ocigenai_config = config['ocigenai']

MODEL_OPTIONS = ['meta.llama-3-70b-instruct', 'cohere.command-r-plus', 'cohere.command-r-16k']

def get_response(user_query, chat_history, model_id, model_kwargs):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=ocigenai_config["service_endpoint"],
        compartment_id=ocigenai_config["compartment_id"],
        model_kwargs=model_kwargs,
        is_stream=True,
    )
    chain = prompt | llm | StrOutputParser() 
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })


if 'chat' not in ss:
    ss.chat = {'model1': [], 'model2': []}
if 'is_good_input' not in ss:
    ss.is_good_input = False


def submit_cb():
    ss.is_good_input = False
    ss.chat['model1'] = []
    ss.chat['model2'] = []
    if ss.mn1 == ss.mn2:
        st.sidebar.warning('models should not be the same')
    else:
        ss.is_good_input = True


def main():
    # Get option values.
    with st.sidebar:
        with st.form('form'):
            st.selectbox('select model name #1', MODEL_OPTIONS, index=0, key='mn1')
            st.selectbox('select model name #2', MODEL_OPTIONS, index=1, key='mn2')
            st.slider('max tokens', value=4000, min_value=1, max_value=4000, step=100, key='maxtoken')
            st.slider('temperature', value=0.5, min_value=0.0, max_value=1.0, step=0.1, key='temperature')
            st.form_submit_button('Submit', on_click=submit_cb)

    if not ss.is_good_input:
        st.stop()

    model1 = ss.mn1
    model2 = ss.mn2
    model_kwargs = {"temperature": ss.temperature, "max_tokens": ss.maxtoken}

    st.title(f"Chat with OCIGenAI {model1} and {model2}")

    left, right = st.columns([1, 1], gap='large')

    with left:
        st.write(f'{model1}')

    with right:
        st.write(f'{model2}')


    with left:
        for message in ss.chat['model1']:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

    with right:
        for message in ss.chat['model2']:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":

        ss.chat['model1'].append(HumanMessage(content=user_query))
        ss.chat['model2'].append(HumanMessage(content=user_query))
        
        with left:
            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                start = time.time()
                response = st.write_stream(get_response(user_query, ss.chat['model1'], model1, model_kwargs))
                end = time.time()
                time_taken = end - start
                st.info(f"""**Duration: :green[{time_taken:.2f} secs]**""")
                ss.chat['model1'].append(AIMessage(content=response))

        with right:
            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                start = time.time()
                response = st.write_stream(get_response(user_query, ss.chat['model2'], model2, model_kwargs))
                end = time.time() 
                time_taken = end - start
                st.info(f"""**Duration: :green[{time_taken:.2f} secs]**""")
                ss.chat['model2'].append(AIMessage(content=response))


if __name__ == '__main__':
    main()