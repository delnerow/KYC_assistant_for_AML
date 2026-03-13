from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
# best practice to use upper-case
entities = Literal["Customer", "Addres", "Account", "Transaction", "Alert", "PEP", "Sanction"]
relations = Literal["LIVES", "HAS_ACCOUNT", "IN_PEP", "MATCHES_SANCTION", "TRIGGERED", "SENT","RECEIVED"]

# define which entities can have which relations
validation_schema = {
        "Customer": ["LIVES", "HAS_ACCOUNT", "IN_PEP", "MATCHES_SANCTION", "TRIGGERED"],
    "Address": ["LIVES"],
        "Account": ["HAS_ACCOUNT"],
        "Transaction": ["SENT", "RECEIVED", "TRIGGERED"],
    "Alert": ["TRIGGERED"],
    "PEP": ["IN_PEP"],
    "Sanction": ["MATCHES_SANCTION"]
}

kg_extractor = SchemaLLMPathExtractor(
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    # if false, allows for values outside of the schema
    # useful for using the schema as a suggestion
    strict=True,
)