from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv
load_dotenv()

import setup_db as _setup
_setup.main()

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
        kwargs = {}
        if os.getenv("NEO4J_DATABASE"):
            kwargs["database"] = os.environ["NEO4J_DATABASE"]
        self.graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            **kwargs,
        )
        self.graph.refresh_schema()

graph = Graph().graph