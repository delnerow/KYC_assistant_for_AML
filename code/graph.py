from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv
load_dotenv()

def singleton(cls):
    """A singleton decorator that stores instances in a dictionary."""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            # Create the single instance if it doesn't exist
            instances[cls] = cls(*args, **kwargs)
        # Always return the stored instance
        return instances[cls]

    return get_instance

@singleton
class Graph:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            database="kyc"
        )
graph = Graph().graph