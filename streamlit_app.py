import boto3
from langchain.llms import Bedrock
import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    MULTI_DF_PREFIX,
    MULTI_DF_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.tools.python.tool import PythonAstREPLTool

import streamlit as st
from streamlit_callback_handler import StreamlitCallbackHandler, LLMThoughtLabeler

def _get_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        include_df_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
        include_df_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_df_head = False

    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
        if include_df_head:
            input_variables += ["df_head"]

    if prefix is None:
        prefix = PREFIX

    tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables,
        format_instructions=FORMAT_INSTRUCTIONS_NEW
    )

    partial_prompt = prompt.partial()
    if "df_head" in input_variables:
        partial_prompt = partial_prompt.partial(
            df_head=str(df.head(number_of_head_rows).to_markdown())
        )
    return partial_prompt, tools

def _get_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import pandas as pd

        pd.set_option("display.max_columns", None)
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        raise ValueError(f"multi-df not supported")
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )

import boto3
from langchain.llms import Bedrock
import datetime
import time

## this uses a cross-account IAM permission 
## that has Bedrock preview access

def getBedrockModel(
    model_id: str,
    model_kwargs: dict,
    role_arn: str = 'arn:aws:iam::123456789012:role/bedrock-cross-account-example'
):
    sts = boto3.client('sts')
    resp = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName='session-'+datetime.datetime.now().isoformat().replace(':','='))
    new_session = boto3.Session(
        aws_access_key_id=resp['Credentials']['AccessKeyId'],
        aws_secret_access_key=resp['Credentials']['SecretAccessKey'],
        aws_session_token=resp['Credentials']['SessionToken'])
    bedrock_region_name = 'us-east-1' ## Choose the appropriate region here
    bedrock = new_session.client('bedrock',region_name=bedrock_region_name)
    # optional sleep, this is a suboptimal way to avoid hitting Bedrock's rate limit
    time.sleep(5)
    anthropicModel = Bedrock(
        client=bedrock,
        model_id=model_id,
        model_kwargs=model_kwargs)
    return anthropicModel

def create_pandas_dataframe_agent_new(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 2,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    agent: BaseSingleActionAgent
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt, tools = _get_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )

import os 
import pandas as pd 
import matplotlib.pyplot as plt 
FORMAT_INSTRUCTIONS_NEW = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
st.set_page_config(page_title="Use Amazon Bedrock to interact with datasets", page_icon="🤖",layout="wide")

with open('./streamlit.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    authenticator.logout('Logout', 'main')
    st.title("🤖 Use Amazon Bedrock to interact with datasets")

    uploaded_file = st.file_uploader(
        "Upload a Data file",
        type=list(file_formats.keys()),
        help="Various File formats are Support",
        on_change=clear_submit,
    )

    if uploaded_file:
        df = load_data(uploaded_file)

    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you? Tip: You should clear conversation via the sidebar if you are asking about a new topic."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        bedrockModel = getBedrockModel(
            'anthropic.claude-v2',
            {
                "max_tokens_to_sample": 2000,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1,
                "stop_sequences": ["\\n\\nHuman:","Question:"],
            },
        )

        agent = create_pandas_dataframe_agent_new(
                bedrockModel, df, verbose=True, max_iterations=50,
                format_instructions=FORMAT_INSTRUCTIONS_NEW)

        with st.chat_message("assistant"):
            # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False, thought_labeler=LLMThoughtLabeler())
            response = agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')