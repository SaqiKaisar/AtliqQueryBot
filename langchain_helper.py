from langchain.chains import create_sql_query_chain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_few_shot_query_chain():
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_host = os.environ["DB_HOST"]
    db_name = os.environ["DB_NAME"]

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3,
    )

    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    few_shots = [
        {
            "Question": "How many t-shirts do we have left for Nike in XS size and white color?",
            "SQLQuery": "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
        },
        {
            "Question": "How much is the total price of the inventory for all S-size t-shirts?",
            "SQLQuery": "SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
        },
        {
            "Question": "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?",
            "SQLQuery": """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
        },
        {
            "Question": "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?",
            "SQLQuery": "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
        },
        {
            "Question": "How many white color Levi's shirt I have?",
            "SQLQuery": "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
        },
        {
            "Question": "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
            "SQLQuery": """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
        },
    ]

    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    # Modify the prompt to return ONLY the SQL query (No Answer, No Execution)
    mysql_prompt = """You are a MySQL expert. Given an input question, generate only the SQL query needed to retrieve the answer.
    Return the asnwer in the following format. It should show the question too:
    Question: {input}\n
    SQLQuery: Query to run with no pre-amble

    - Never query for all columns from a table; query only the necessary columns.
    - Wrap column names in backticks (`).
    - Ensure column names exist in the provided schema.
    - Use `CURDATE()` for "today"-related queries.
    - Limit query results to {top_k} rows unless otherwise specified.

    {table_info}
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix="",
        input_variables=["input", "table_info", "top_k"],  # Used in prefix
    )

    # Use create_sql_query_chain instead of SQLDatabaseChain
    chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)

    return llm, db, chain


def format_answer(llm, response):
    query_format = f"write this value {response} as just a number or format it properly if it is a list return it as a comma seperated list if a list. Just give me the formatted value. I dont need anything extra"
    temp = llm.invoke(query_format)
    answer = "Answer: " + (str(temp.content))
    return answer


def run_query_on_db(query, db, chain, llm):
    try:
        original_sql_query = chain.invoke({"question": query})
        sql_query = original_sql_query
        if isinstance(sql_query, str):
            # Use partition to split and include "SELECT"
            _, separator, after_select = sql_query.partition("SELECT")
            sql_query = separator + " " + after_select.replace("\n", " ").strip()

        response = db.run(sql_query)
        answer = original_sql_query + "\n" + format_answer(llm, response)

        return answer
    except:
        message = "I cannot answer the question with my knowledgebase. Either the information does not exist in the database or some mistakes in the query."
        return message
