"""
setup_db.py — Ensure the Neo4j database is populated with KYC data.

Run from the code/ directory:
    python setup_db.py

On Neo4j Aura (Free/Professional), do NOT set NEO4J_DATABASE — the driver
will route to the correct database automatically via the connection URI.
On local/Enterprise Neo4j, optionally set NEO4J_DATABASE=<name>.

What it does:
    1. Connects and verifies credentials.
    2. Optionally creates a named database (Enterprise only; skipped on Aura).
    3. Checks if the database already has data — skips load if so.
    4. Loads all CSVs from ./data/ in the correct order
       (nodes first, then relationships).
"""

import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

load_dotenv()

URI      = os.environ["NEO4J_URI"]
USER     = os.environ["NEO4J_USERNAME"]
PASSWORD = os.environ["NEO4J_PASSWORD"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BATCH    = 500

# None = let the driver use the default database (correct for Aura).
# Set NEO4J_DATABASE env var only for local/Enterprise multi-database setups.
_DB = os.getenv("NEO4J_DATABASE") or None


# ── utilities ─────────────────────────────────────────────────────────────────
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_csv(filename: str) -> list[dict]:
    df = pd.read_csv(os.path.join(DATA_DIR, filename)).fillna("")
    return df.to_dict("records")


def session(driver):
    """Return a session, routing to the default db on Aura or _DB if set."""
    kwargs = {"database": _DB} if _DB else {}
    return driver.session(**kwargs)


# ── step 1: try to create named database (Enterprise / local only) ────────────
def try_create_database(driver):
    if not _DB:
        print("  No NEO4J_DATABASE set — using default database (Aura mode).")
        return
    print(f"  Ensuring database '{_DB}' exists...")
    try:
        with driver.session(database="system") as s:
            s.run(f"CREATE DATABASE {_DB} IF NOT EXISTS WAIT")
        print(f"  ✓ Database '{_DB}' is ready.")
    except ClientError as e:
        if any(w in str(e).lower() for w in ("not supported", "unsupported", "permission", "not allowed")):
            print(f"  ⚠  CREATE DATABASE not supported on this tier — skipping.")
        else:
            raise


# ── step 2: check if populated ───────────────────────────────────────────────
def is_empty(driver) -> bool:
    with session(driver) as s:
        return s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"] == 0


# ── step 3: indexes ───────────────────────────────────────────────────────────
INDEXES = [
    "CREATE INDEX customer_id    IF NOT EXISTS FOR (n:Customer)    ON (n.id)",
    "CREATE INDEX account_id     IF NOT EXISTS FOR (n:Account)     ON (n.id)",
    "CREATE INDEX address_id     IF NOT EXISTS FOR (n:Address)     ON (n.id)",
    "CREATE INDEX transaction_id IF NOT EXISTS FOR (n:Transaction) ON (n.id)",
    "CREATE INDEX pep_id         IF NOT EXISTS FOR (n:PEP)         ON (n.id)",
    "CREATE INDEX alert_id       IF NOT EXISTS FOR (n:Alert)       ON (n.id)",
    "CREATE INDEX sanction_id    IF NOT EXISTS FOR (n:Sanction)    ON (n.id)",
]

def create_indexes(driver):
    print("Creating indexes...")
    with session(driver) as s:
        for idx in INDEXES:
            s.run(idx)
    print("  ✓ Done.")


# ── step 4: node loaders ──────────────────────────────────────────────────────
def _load(driver, csv_file, cypher, label):
    rows = load_csv(csv_file)
    print(f"Loading {len(rows):,} {label}...")
    with session(driver) as s:
        for batch in chunks(rows, BATCH):
            s.run(cypher, rows=batch)
    print("  ✓ Done.")


def load_customers(driver):
    _load(driver, "customers.csv", """
    UNWIND $rows AS r
    MERGE (c:Customer {id: r.id})
    SET c.name               = r.name,
        c.email              = r.email,
        c.phone              = r.phone,
        c.date_of_birth      = r.date_of_birth,
        c.nationality        = r.nationality,
        c.type               = r.type,
        c.kyc_status         = r.kyc_status,
        c.risk_level         = r.risk_level,
        c.tax_id             = r.tax_id,
        c.registration_number = r.registration_number,
        c.created_date       = r.created_date
    """, "customers")


def load_accounts(driver):
    _load(driver, "accounts.csv", """
    UNWIND $rows AS r
    MERGE (a:Account {id: r.id})
    SET a.account_number = r.account_number,
        a.type           = r.type,
        a.currency       = r.currency,
        a.balance        = toFloat(r.balance),
        a.status         = r.status,
        a.opened_date    = r.opened_date
    """, "accounts")


def load_addresses(driver):
    _load(driver, "addresses.csv", """
    UNWIND $rows AS r
    MERGE (a:Address {id: r.id})
    SET a.street      = r.street,
        a.city        = r.city,
        a.state       = r.state,
        a.postal_code = r.postal_code,
        a.country     = r.country
    """, "addresses")


def load_transactions(driver):
    _load(driver, "transactions.csv", """
    UNWIND $rows AS r
    MERGE (t:Transaction {id: r.id})
    SET t.amount     = toFloat(r.amount),
        t.currency   = r.currency,
        t.date       = date(r.date),
        t.type       = r.type,
        t.risk_score = toFloat(r.risk_score)
    """, "transactions")


def load_peps(driver):
    _load(driver, "peps.csv", """
    UNWIND $rows AS r
    MERGE (p:PEP {id: r.id})
    SET p.list_name     = r.list_name,
        p.position      = r.position,
        p.country       = r.country,
        p.risk          = r.risk,
        p.verified_date = r.verified_date
    """, "PEPs")


def load_alerts(driver):
    _load(driver, "alerts.csv", """
    UNWIND $rows AS r
    MERGE (a:Alert {id: r.id})
    SET a.type         = r.type,
        a.status       = r.status,
        a.severity     = r.severity,
        a.created_date = r.created_date
    """, "alerts")


def load_sanctions(driver):
    _load(driver, "sanctions.csv", """
    UNWIND $rows AS r
    MERGE (s:Sanction {id: r.id})
    SET s.list_name   = r.list_name,
        s.entity_name = r.entity_name,
        s.match_score = toFloat(r.match_score),
        s.match_date  = r.match_date
    """, "sanctions")


# ── step 5: relationship loaders ──────────────────────────────────────────────
def load_relationships(driver):
    _load(driver, "customer_accounts.csv", """
    UNWIND $rows AS r
    MATCH (c:Customer {id: r.customer_id})
    MATCH (a:Account  {id: r.account_id})
    MERGE (c)-[:HAS_ACCOUNT]->(a)
    """, "HAS_ACCOUNT relationships")

    _load(driver, "customer_addresses.csv", """
    UNWIND $rows AS r
    MATCH (c:Customer {id: r.customer_id})
    MATCH (a:Address  {id: r.address_id})
    MERGE (c)-[:LIVES_AT]->(a)
    """, "LIVES_AT relationships")

    _load(driver, "transaction_sent.csv", """
    UNWIND $rows AS r
    MATCH (a:Account     {id: r.account_id})
    MATCH (t:Transaction {id: r.transaction_id})
    MERGE (a)-[:SENT]->(t)
    """, "SENT relationships")

    _load(driver, "transaction_received.csv", """
    UNWIND $rows AS r
    MATCH (t:Transaction {id: r.transaction_id})
    MATCH (a:Account     {id: r.account_id})
    MERGE (t)-[:RECEIVED]->(a)
    """, "RECEIVED relationships")

    _load(driver, "customer_sanctions.csv", """
    UNWIND $rows AS r
    MATCH (c:Customer {id: r.customer_id})
    MATCH (s:Sanction {id: r.sanction_id})
    MERGE (c)-[:MATCHES_SANCTION]->(s)
    """, "MATCHES_SANCTION relationships")

    _load(driver, "customer_pep.csv", """
    UNWIND $rows AS r
    MATCH (c:Customer {id: r.customer_id})
    MATCH (p:PEP      {id: r.pep_id})
    MERGE (c)-[:IN_PEP]->(p)
    """, "IN_PEP relationships")

    _load(driver, "customer_alerts.csv", """
    UNWIND $rows AS r
    MATCH (c:Customer {id: r.customer_id})
    MATCH (a:Alert    {id: r.alert_id})
    MERGE (c)-[:TRIGGERED]->(a)
    """, "TRIGGERED (customer) relationships")

    _load(driver, "transaction_alerts.csv", """
    UNWIND $rows AS r
    MATCH (t:Transaction {id: r.transaction_id})
    MATCH (a:Alert       {id: r.alert_id})
    MERGE (t)-[:TRIGGERED]->(a)
    """, "TRIGGERED (transaction) relationships")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nConnecting to {URI}...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  ✓ Connected.\n")

    try_create_database(driver)

    if not is_empty(driver):
        with session(driver) as s:
            cnt = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        print(f"Database already has {cnt:,} nodes — skipping load.")
        driver.close()
        return

    print(f"\nDatabase is empty. Loading data from {DATA_DIR}/\n")
    create_indexes(driver)

    load_customers(driver)
    load_accounts(driver)
    load_addresses(driver)
    load_transactions(driver)
    load_peps(driver)
    load_alerts(driver)
    load_sanctions(driver)

    load_relationships(driver)

    driver.close()
    print("\n✅ Setup complete. Database is ready.")


if __name__ == "__main__":
    main()
