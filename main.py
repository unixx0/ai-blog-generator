from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Any, Literal
from langchain_groq import ChatGroq
import operator
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.types import Send
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from huggingface_hub import InferenceClient
import pypandoc
from logger import get_logger


logger= get_logger()

try:
    if not os.getenv('tavily'):
        raise KeyError("key not found")
    os.environ["TAVILY_API_KEY"] = os.getenv('tavily')


except Exception as e:
    logger.critical(f"failed to set tavily api key: {e}")
    raise    #for termination of program

#Pydantic model required for workflow
class Task(BaseModel):
    id: int
    title: str= Field(description= "The title of the section")
    brief: str= Field(description= "What to cover on the given section")

class Plan(BaseModel):
    blog_title: str= Field(description= "The Title of the blog")
    tasks: list[Task]

class router_task(BaseModel):
    questions: list[str]= Field(description= "The questions generated", min_length= 5, max_length= 7)
    need_web: bool
    
class ImageSpec(BaseModel):
    placeholder:str= Field(..., description= "The name of placeholder. eg: [[IMAGE_1]]")
    save_path: str= Field(..., description= "The path where image will be saved. Strictly like:- images/image1.png, images/image2.png and so on")
    prompt: str= Field(..., description= "Prompt to send to Image model to generate required image")
    size: Literal['1024x1024', '1024x1536', '1536x1024']= Field(default= '1024x1024', description= "The size of image")
    alt: str
    caption: str
    quality: Literal["low", 'medium', 'high']= Field(default= 'medium', description= "The quality of image to generate")

class GlobalImageplan(BaseModel):
    md_with_placeholder: str= Field(description= "md consisting the placeholders (eg. '[[IMAGE_1]]' ) where required")
    images: list[ImageSpec]



#States of Parent graph and subgraph
class State(TypedDict):
    topic: str
    sections: Annotated[list[str], operator.add]
    plan: Plan
    final:str
    router_decision: router_task
    web_result: list[dict[str, Any]]
    title:str
    
    
class SubgraphState(TypedDict):
    title: str
    sections: list[str]
    md: str
    intermediate_md: GlobalImageplan
    md_with_image: str
    topic:str
    img_path: list[str]
    pdf:bool
    name: str

try:
    if not os.getenv('groq2'):
        raise KeyError("Failed to set the groq llm")
    llm= ChatGroq(
        model= "llama-3.3-70b-versatile",
        api_key= os.getenv('groq2')
    )
except Exception as e:    #gets what raise throws as keyerror falls under Exception Class
    logger. exception(e)


try:
    #here the programmer hasnt manually raised an error so the program automatically finds the exception if it arrises
    hf_llm= HuggingFaceEndpoint(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature= 0.6,
        huggingfacehub_api_token= os.getenv('test')
    )

    llm2= ChatHuggingFace(
        llm= hf_llm
    )

    client = InferenceClient(
        provider="nscale",
        api_key=os.getenv('test'),
    )
except KeyError as e:
    logger.exception(e)

web= TavilySearch(max_results= 1)

#defining subgraph
def concact_md(state: SubgraphState):
    logger.info("Concat_md is executing")
    title= state['title']. strip()
    body= '\n\n'.join(state["sections"]).strip()

    final_md= f"# {title}\n\n{body}"

    return {'md': final_md}

def placeholder_generator(state: SubgraphState):
    logger.info("Placeholder is being generated for the blog")
    md= state['md']

    result= llm.with_structured_output(schema= GlobalImageplan).invoke([
        SystemMessage(content= '''From the given markdown file, insert image placeholders WITHIN the content where images would help visualize concepts.
        
        CRITICAL RULES:
        - Return the COMPLETE markdown with placeholders embedded in the text
        - Do NOT change any existing content, only insert placeholders
        - Insert placeholders like [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] INSIDE paragraphs or after relevant sentences
        - Maximum 3 placeholders total
        - Place placeholders immediately after the paragraph they illustrate, NOT at section headers
        - Save paths must be: images/image1.png, images/image2.png, images/image3.png
        - Generate technical/diagram-focused prompts (flowcharts, architecture diagrams, technical illustrations)
        - Prompts should be detailed and specific for accurate image generation
        
        Example format:
        "## Section Title
        Some text explaining a concept.
        
        [[IMAGE_1]]
        
        More text continuing..."'''),

        HumanMessage(content= f"Markdown File: {md}")
    ])

    return {'intermediate_md': result}

def image_generator(state: SubgraphState):
    logger.info("Image is being generated for the blog")
    images= state['intermediate_md'].images
    os.makedirs("images", exist_ok= True)
    img_path: list[str]= []

    for i, info in enumerate(images, start= 1):
        result= client.text_to_image (prompt= f"""Generate image on the basis of given information
    prompt: {info.prompt},
    size: {info.size}, 
    quality: {info.quality}""",
    model= "stabilityai/stable-diffusion-xl-base-1.0"
)
        
        result.save(f"images/image{i}.png")
        img_path.append(f"images/image{i}.png")
        
        
    logger.info("All the images saved to images directory")
    return {'img_path': img_path}

def merge_image_md(state: SubgraphState):
    logger.info("placeholders are getting replaced by images")
    markdown= state['intermediate_md'].md_with_placeholder
    img_info= state['intermediate_md'].images
    img_path= state['img_path']

    for i, x in enumerate(img_info):
        markdown= markdown.replace(x.placeholder, f'![{x.alt}]({img_path[i]} "{x.caption}")')

    name= state['topic'].lower().strip().replace(" ", "_")
    with open(f"{name}.md", 'w') as file:
        file.write(markdown)

    logger.info("Markdown Saved")
    print(f"Markdown successfully stored as {name}.md")


    logger.info("HITL Initialized")
    decision= interrupt("Do you want to generate its pdf too? Answer in yes or no only")

    if decision.strip().lower()== "yes":
        os.makedirs("PDF's", exist_ok= True)
        pypandoc.convert_file(
            source_file= f"{name}.md",
            to= "pdf",
            outputfile= os.path.join("PDF's", f"{name}.pdf")
        )
        print("PDF created")
    elif decision.strip().lower()== "no":
        print("You choosed not to generate pdf")

    else:
        print("You must choose between yes or no")
        
    return {"md_with_image": markdown}

subgraph= StateGraph(state_schema= SubgraphState)

subgraph.add_node("merge_image_md", merge_image_md)
subgraph.add_node("image_generator", image_generator)
subgraph.add_node("placeholder_generator", placeholder_generator)
subgraph.add_node("concat_md", concact_md)

subgraph.add_edge(START, 'concat_md')
subgraph.add_edge("concat_md", "placeholder_generator")
subgraph.add_edge("placeholder_generator", "image_generator")
subgraph.add_edge("image_generator", "merge_image_md")
subgraph.add_edge("merge_image_md", END)

subworkflow= subgraph.compile()

#Defining the Parent Graph
def router(state: State):
    logger.info('Workflow Started')
    logger.info('Router initialized')
    system= """Decide whether the given topic need web search or not. 
    need_web= True; if the topic requires web serch. 
    If it needs web search then generate some questions regarding to that topic.
    Make the questions diverse and donot repeat same kind of questions.
    
    need_web= False; means the questions donot require web search.
"""


    result= llm.with_structured_output(schema= router_task).invoke([
        SystemMessage(content= system), 
        HumanMessage(content= state['topic'])
    ])

    #logger.info(f"Router choice: {state['router_decision'].need_web}")
    return {'router_decision': result
            }

def decision(state: State) -> bool:
    if state['router_decision'].need_web:
        return True
    else:
        return False

def web_search(state: State):
    """Search the given questions in web"""

    web_result: list[dict]= []  #initializing variable by assigning property
    
    for question in state['router_decision'].questions:
        result= web.invoke(input= {"query": question})
        web_result.extend(result['results'])
        
    return {'web_result': web_result}

def orchestrator (state: State):

    logger.info("Orchestrator Initialised")
    system=  "You are a excellent Blog Writter. Now create a detailed blog plan consisting of around 5-7 section on the given topic"
    user= f"Topic: {state['topic']}"


    result= llm.with_structured_output(schema= Plan).invoke([
        SystemMessage(content= system), HumanMessage(content= user)

    ])

    return {
        "plan": result,   #returns Plan object
          "title": result.blog_title  
    }

def fanout(state: State): 
    if state.get('web_result', ""):    #is there exist web_result key in state then it returns its value else it will be empty string
         # it is an conditional edge 
        return[Send("workers", {"topic": state['topic'], 'plan': state['plan'], 'task': task, 'need_web': state['router_decision'].need_web, 'additional_info': state['web_result']}) for task in state['plan']. tasks]  #send is used to send each task object to a workers and the no. of workers run parallelly
    else:
        return[Send("workers", {"topic": state['topic'], 'plan': state['plan'], 'task': task, 'need_web': state['router_decision'].need_web}) for task in state['plan']. tasks]

def workers(payload: dict): #gets what fanout returned. No of worker node created= no of send objects returned by fanout and each worker node receives what it send

    logger.info("Workers Initialized")
    logger.info(f"Router Decision {payload['need_web']}")
    topic= payload['topic']
    plan= payload['plan']
    task= payload['task']
    decision= payload['need_web']
    if payload.get('additional_info', ''):
        additional_info= payload['additional_info']
    
    

    if decision:

        result= llm2.invoke(input=[
            SystemMessage(content='''Write a clean markdown section and return only the section content in markdown. 
                
            RULES:
            - Take consideration of given additional information and ONLY use the contents that is related to title of sections. 
            - Include hyperlinks refering the url if it used it. Use examples if required.
            - Only use the information from trusted sources and donot use the information from same url again and again.'''),
            HumanMessage(content= f'''
                                blog topic: {topic},
                                title: {plan.blog_title}
                                section title: {task.title}
                                section brief: {task.brief}
                                additional info: {additional_info}
    ''')
        ])

    else:
        result= llm.invoke(input=[
            SystemMessage(content= "Write a clean markdown section and return only the section content in markdown. Also use examples if required"),
            HumanMessage(content= f'''
                                blog topic: {topic},
                                title: {plan.blog_title}
                                section title: {task.title}
                                section brief: {task.brief}
                                
    ''')
        ])



    return {"sections": [result.content]}

def reducer(state: State):
    logger.info("reducer initialized")
    #Invoking the subgraph here
    logger.info("Subgraph Invoked")
    
    result= subworkflow.invoke(input= {
        'title': state['title'],
        'sections': state['sections'],
        'topic': state['topic']
    })

    return {'final': result['md_with_image']}


graph= StateGraph(state_schema= State)
graph.add_node('router', router)
graph.add_node("orchestrator", orchestrator)
graph.add_node("workers", workers)
graph.add_node('reducer', reducer)
graph.add_node('web_search', web_search)


graph.add_edge(START, "router")
graph.add_conditional_edges("router", decision, {True: 'web_search', False: 'orchestrator'})
graph.add_edge('web_search','orchestrator')
graph.add_conditional_edges("orchestrator", fanout, ['workers'])
graph.add_edge('workers', "reducer")
graph.add_edge("reducer", END)

checkpointer= InMemorySaver()
workflow= graph.compile(checkpointer= checkpointer)
config= {"configurable": {"thread_id": "1"}}

topic= input("Enter the topic for your blog generation: ")
logger.info("Parent Graph invoked")
result= workflow.invoke(input={'topic': topic}, config= config)

print(result['__interrupt__'][0].value) #Value inside the interrupt function 

choice= input("Enter: ")

result= workflow.invoke(Command(resume= choice), config= config)





    
   


