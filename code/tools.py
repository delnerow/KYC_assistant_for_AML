
from langchain.tools import tool
from code.graph import graph


def execute_cypher_query( cypher, params=None):
    """Executes a Cypher query against the Neo4j graph database and returns the results."""
    return graph.query(cypher, params)

@tool
def get_id_by_name( name):
    """Retrieves the ID of a node based on its name. Usually names on the node have capitalized first letter, so we can try to capitalize firsts letter of the name if not found."""
    cypher = """
    MATCH (c: Customer {name: $name})
    RETURN c.id AS id
    """
    result = execute_cypher_query( cypher, params={"name": name})
    return result[0]['id'] if result else None

@tool
def get_customer_accounts( customer_id):
    """ Retrieves the accounts associated with a customer.
    Args:       customer_id: The ID of the customer to retrieve accounts for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)
    RETURN a
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})

@tool
def get_customer_sanctions( customer_id):
    """ Retrieves the sanctions associated with a customer.
    Args:       customer_id: The ID of the customer to retrieve sanctions for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})-[:MATCHES_SANCTION]->(s:Sanction)
    RETURN s
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})

@tool
def get_customer_by_account( account_id):
    """ Retrieves the customer associated with a given account ID.
    Args:
        account_id: The ID of the account to find the associated customer for.
    """
    cypher = """
    MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account {id: $account_id})
    RETURN c
    """
    return execute_cypher_query( cypher, params={"account_id": account_id})

def get_customer_info( customer_id):
    """ Retrieves basic information about a customer, such as name, email, phone, KYC status, risk level, date of birth (if individual), and nationality.
    Args:
        customer_id: The ID of the customer to retrieve information for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})
    RETURN c.name AS name, c.dob AS dob, c.nationality AS nationality, c.type AS type, c.email AS email, c.phone AS phone, c.date_of_birth AS date_of_birth, c.kyc_status AS kyc_status, c.registration_number AS registration_number, c.risk_level AS risk_level
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})

@tool
def get_customer_profile( customer_id):
    """
    Retrieves a customer's profile, including their accounts, transactions, receivers, sanctions, and PEPs.
    Args:
        customer_id: The ID of the customer to retrieve the profile for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})
    OPTIONAL MATCH (c)-[:HAS_ACCOUNT]->(a:Account)
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)-[:RECEIVED]->(r:Account)
    OPTIONAL MATCH (c)-[:MATCHES_SANCTION]->(s:Sanction)
    OPTIONAL MATCH (c)-[:IN_PEP]->(p:PEP)
    RETURN c, collect(DISTINCT a) AS accounts, collect(DISTINCT t) AS transactions,
           collect(DISTINCT r) AS receivers, collect(DISTINCT s) AS sanctions,
           collect(DISTINCT p) AS peps
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})

@tool
def get_customer_risk_summary( customer_id):
    """Generates a risk summary for a customer, including counts of accounts, transactions, sanctions, and PEPs.
    Args:
    customer_id: The ID of the customer to generate the risk summary for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})
    OPTIONAL MATCH (c)-[:HAS_ACCOUNT]->(a:Account)
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    OPTIONAL MATCH (c)-[:MATCHES_SANCTION]->(s:Sanction)
    OPTIONAL MATCH (c)-[:IN_PEP]->(p:PEP)
    RETURN c.name AS name, c.dob AS dob, c.nationality AS nationality,
           count(DISTINCT a) AS num_accounts, count(DISTINCT t) AS num_transactions,
           count(DISTINCT s) AS num_sanctions, count(DISTINCT p) AS num_peps
    """
    return execute_cypher_query(cypher, params={"customer_id": customer_id})

def get_account_info( account_id):
    """ Retrieves basic information about an account, such as account number, type, balance, and associated customer.
    Args:
        account_id: The ID of the account to retrieve information for.
    """
    cypher = """
    MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account {id: $account_id})
    RETURN a.account_number AS account_number, a.type AS type, a.balance AS balance, c.name AS customer_name
    """
    return execute_cypher_query( cypher, params={"account_id": account_id})

def filter_customers( list_of_filters:dict):
    """ Filters customers based on specified criteria such as nationality, kyc_status (pending, approved, rejected), risk_level(low, medium, high),date_of_birth, type(individual, business), phone (substrings), email (substrings).
    Args:
        list_of_filters: A list of filter criteria.
    """
    # Implementation for filtering customers
    cypher = """
MATCH (c:Customer)
WHERE (c.nationality is not NULL or c.nationality is $nationality) AND
      (c.kyc_status is not NULL or c.kyc_status = $kyc_status) AND
      (c.risk_level is not NULL or c.risk_level = $risk_level) AND
      (c.date_of_birth is not NULL or c.date_of_birth = $date_of_birth) AND
      (c.type is not NULL or c.type = $type) AND
      (c.phone is not NULL or c.phone = $phone) AND
      (c.email is not NULL or c.email = $email)
      RETURN c
"""
    return execute_cypher_query( cypher, params=list_of_filters)

@tool
def extract_customer_transactions_period( customer_id, start_date=None, end_date=None):
    """Extracts transactions for a customer within a specified date range. If no start_date and end_date are provided, it will extract all transactions for the customer.
    Args:       
        customer_id: The ID of the customer to extract transactions for.
        start_date: The start date of the period to extract transactions from (inclusive).
        end_date: The end date of the period to extract transactions from (inclusive).
    """
    if not start_date or not end_date:
        cypher = """
        MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
        RETURN t
        """
        return execute_cypher_query( cypher, params={"customer_id": customer_id})
    else:
        cypher = """
        MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)
        WHERE t.date >= $start_date AND t.date <= $end_date
        RETURN t
        """
    return execute_cypher_query( cypher, params={"customer_id": customer_id, "start_date": start_date, "end_date": end_date})

@tool
def trace_shared_accounts( customer_id=None):
    """ Traces shared accounts between customers. A customer might be specified or it will iterate over all customers.
    Args:
        customer_id: The ID of the customer to trace shared accounts for. If None, it will trace for all customers.
    """
    if customer_id:
        cypher = """
        MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)<-[:HAS_ACCOUNT]-(other:Customer)
        WHERE other.id <> c.id
        RETURN DISTINCT other AS shared_account_customer
        """
        return execute_cypher_query( cypher, params={"customer_id": customer_id})
    else:
        cypher = """
        MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)<-[:HAS_ACCOUNT]-(other:Customer)
        WHERE other.id <> c.id
        RETURN DISTINCT c AS customer, other AS shared_account_customer
        """
        return execute_cypher_query(cypher, params={"customer_id": customer_id})
    
@tool
def trace_shared_addresses(customer_id=None):
    """ Traces shared addresses between customers. A customer might be specified or it will iterate over all customers. 
    Args:
        customer_id: The ID of the customer to trace shared addresses for. If None, it will trace for all customers.
    """
    if customer_id:
        cypher = """
        MATCH (c:Customer {id: $customer_id})-[:LIVES_AT]->(a:Address)<-[:LIVES_AT]-(other:Customer)
        WHERE other.id <> c.id
        RETURN DISTINCT other AS shared_address_customer
        """
        return execute_cypher_query(cypher, params={"customer_id": customer_id})
    else:
        cypher = """
       MATCH (c:Customer)-[:LIVES_AT]->(a:Address)<-[:LIVES_AT]-(other:Customer)
        WHERE other.id <> c.id
        RETURN DISTINCT c AS customer, other AS shared_address_customer
        """
        return execute_cypher_query(cypher, params={"customer_id": customer_id})
@tool
def trace_shared_phone_numbers( customer_id=None):
    """ Traces shared phone numbers between customers. A customer might be specified or it will iterate over all customers.
    Args:
        customer_id: The ID of the customer to trace shared phone numbers for. If None, it will trace for all customers.
    """
    if customer_id:
        cypher = """
        MATCH (c:Customer {id: $customer_id})
        WITH c, c.phone as part
        MATCH (other:Customer)
        WHERE other.id <> c.id AND part IN other.phone
        RETURN DISTINCT other
        """
        return execute_cypher_query(cypher, params={"customer_id": customer_id})
    else:
        cypher = """
        MATCH (c:Customer)
        WITH c, c.phone as part
        MATCH (other:Customer)
        WHERE other.id <> c.id AND part IN other.phone
        RETURN DISTINCT c AS customer, other AS shared_phone_customer
        """
        return execute_cypher_query(cypher, params={"customer_id": customer_id})
    
@tool
def find_mutual_counterparties( customer_id):
    """ Finds mutual counterparties between customers based on transactions.
        Args:
        customer_id: The ID of the customer to find mutual counterparties for. 
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED]->(r:Account)<-[:HAS_ACCOUNT]-(other:Customer)
    RETURN DISTINCT other
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})
    

@tool
def summarize_customer_risk( customer_id):
    """
    Generates a comprehensive risk summary for a customer, combining account activity, relational links, transaction behavior, and compliance signals such as sanctions, PEPs, and alerts.
    Args:        customer_id: The ID of the customer to summarize risk for.
    """
    cypher = """
    MATCH (c:Customer {id: $customer_id})
    OPTIONAL MATCH (c)-[:HAS_ACCOUNT]->(a:Account)
    OPTIONAL MATCH (a)-[:SENT]->(t:Transaction)
    OPTIONAL MATCH (c)-[:MATCHES_SANCTION]->(s:Sanction)
    OPTIONAL MATCH (c)-[:IN_PEP]->(p:PEP)
    OPTIONAL MATCH (c)-[:TRIGGERED]->(al:Alert)
    RETURN c.name AS name, c.nationality AS nationality,
           count(DISTINCT a) AS num_accounts, count(DISTINCT t) AS num_transactions,
           count(DISTINCT s) AS num_sanctions, count(DISTINCT p) AS num_peps, count(DISTINCT al) AS num_alerts
    """
    return execute_cypher_query( cypher, params={"customer_id": customer_id})


@tool
def get_graph_schema( customer_id=None):
    """ Retrieves the schema of the graph database, including node labels, relationship types, and property keys. This can help understand the structure of the graph and how to query it effectively. """
    
    cypher = """
    CALL db.schema.visualization()
    """
    return execute_cypher_query(cypher, params={"customer_id": customer_id})

@tool
def format_graph_results(results: list):
    """
    Formats Neo4j query results into readable triples
    """

    if not results:
        return "No relationships found."

    lines = []

    for row in results:
        source = row.get("source")
        relation = row.get("relation")
        target = row.get("target")

        lines.append(f"{source} --{relation}--> {target}")

    return "\n".join(lines)
@tool
def trace_shared_emails( customer_id=None):
    """ Traces shared emails between customers. A customer might be specified or it will iterate over all customers.
    Args:
        customer_id: The ID of the customer to trace shared emails for. If None, it will trace for all customers.
    """
    if customer_id:
        cypher = """
        MATCH (c:Customer {id: $customer_id})
        WITH c, c.email as part
        MATCH (other:Customer)
        WHERE other.id <> c.id AND part IN other.email
        RETURN DISTINCT other
        """
        return execute_cypher_query( cypher, params={"customer_id": customer_id})
    else:
        cypher = """
        MATCH (c:Customer)
        WITH c, c.email as part
        MATCH (other:Customer)
        WHERE other.id <> c.id AND part IN other.email
        RETURN DISTINCT c AS customer, other AS shared_email_customer
        """
        return execute_cypher_query( cypher, params={"customer_id": customer_id})

set_tools = [get_account_info,get_customer_by_account, format_graph_results, extract_customer_transactions_period, get_id_by_name, trace_shared_phone_numbers, trace_shared_emails, trace_shared_addresses, find_mutual_counterparties, summarize_customer_risk, get_graph_schema]