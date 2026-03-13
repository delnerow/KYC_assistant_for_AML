from faker import Faker
import pandas as pd
import random
import uuid

fake = Faker()

NUM_CUSTOMERS = 1000
MAX_ACCOUNTS = 3
NUM_TRANSACTIONS = 2000

# distributions
nationalities = ["US","US","US","US","CA","UK","DE","FR","BR","IN"]
currencies = ["USD","USD","USD","EUR","GBP"]
account_types = ["checking","savings","brokerage"]
account_status = ["active","active","active","frozen","closed"]
kyc_status = ["verified","verified","verified","pending","rejected"]
risk_levels = ["low","low","low","medium","medium","high"]

customers = []
accounts = []
addresses = []
transactions = []

cust_acc_rel = []
cust_addr_rel = []
sent_rel = []
recv_rel = []

account_ids = []

pep_list_names = [
    "FATF",
    "WorldCheck",
    "OFAC",
    "EU Sanctions",
    "UN Sanctions"
]

pep_positions = [
    "Head of State",
    "Minister",
    "Member of Parliament",
    "Senior Military Officer",
    "State Owned Enterprise Executive"
]

pep_risk = ["high","high","medium","low"]

alert_types = [
    "AML_SUSPICIOUS_TRANSACTION",
    "SANCTIONS_MATCH",
    "PEP_MATCH",
    "STRUCTURING",
    "LARGE_TRANSACTION"
]

alert_status = ["open","open","closed","under_review"]

alert_severity = ["low","medium","high","critical"]

sanction_lists = [
    "OFAC",
    "EU Sanctions",
    "UN Sanctions",
    "UK HMT",
    "DFAC",
    "WorldCheck"
]
sanctions = []

cust_sanction_rel = []
acct_sanction_rel = []
peps = []
alerts = []

cust_pep_rel = []
cust_alert_rel = []
tx_alert_rel = []
# ---------------------
# Generate Customers + Address + Accounts
# ---------------------

for _ in range(NUM_CUSTOMERS):

    cid = str(uuid.uuid4())

    ctype = random.choices(
        ["individual","corporate"],
        weights=[0.85,0.15]
    )[0]

    nationality = random.choice(nationalities)

    tax_id = fake.ssn() if nationality=="US" else fake.bothify("??########")

    reg = ""
    dob = ""

    if ctype == "corporate":
        reg = fake.bothify("REG-######")
        name = fake.company()
    else:
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
        name = fake.name()

    customers.append({
        "id":cid,
        "name":name,
        "email":fake.email(),
        "phone":fake.phone_number(),
        "date_of_birth":dob,
        "nationality":nationality,
        "type":ctype,
        "kyc_status":random.choice(kyc_status),
        "risk_level":random.choice(risk_levels),
        "tax_id":tax_id,
        "registration_number":reg,
        "created_date":fake.date_between("-5y","today")
    })

    # address node
    addr_id = str(uuid.uuid4())

    addresses.append({
        "id":addr_id,
        "street":fake.street_address(),
        "city":fake.city(),
        "state":fake.state(),
        "postal_code":fake.postcode(),
        "country":fake.country()
    })

    cust_addr_rel.append({
        "customer_id":cid,
        "address_id":addr_id
    })

    # accounts
    for _ in range(random.randint(1,MAX_ACCOUNTS)):

        aid = str(uuid.uuid4())

        accounts.append({
            "id":aid,
            "account_number":fake.bban(),
            "type":random.choice(account_types),
            "currency":random.choice(currencies),
            "balance":round(random.uniform(0,500000),2),
            "status":random.choice(account_status),
            "opened_date":fake.date_between("-5y","today")
        })

        account_ids.append(aid)

        cust_acc_rel.append({
            "customer_id":cid,
            "account_id":aid
        })


# ---------------------
# Generate Transactions
# ---------------------

for _ in range(NUM_TRANSACTIONS):

    tid = str(uuid.uuid4())

    sender = random.choice(account_ids)
    receiver = random.choice(account_ids)

    currency = random.choice(currencies)

    transactions.append({
        "id":tid,
        "amount":round(random.uniform(10,20000),2),
        "currency":currency,
        "date":fake.date_between("-2y","today"),
        "type":random.choice(["debit","credit"]),
        "risk_score":round(random.uniform(0,1),2)
    })

    sent_rel.append({
        "account_id":sender,
        "transaction_id":tid
    })

    recv_rel.append({
        "account_id":receiver,
        "transaction_id":tid
    })

# ---------------------
# Generate Sanctions
# ---------------------
for c in customers:

    if random.random() < 0.01:

        sid = str(uuid.uuid4())

        sanction = {
            "id": sid,
            "list_name": random.choice(sanction_lists),
            "entity_name": c["name"],
            "match_score": round(random.uniform(80,100),2),
            "match_date": fake.date_between("-2y","today")
        }

        sanctions.append(sanction)

        cust_sanction_rel.append({
            "customer_id": c["id"],
            "sanction_id": sid
        })
# ---------------------
# Generate PEPs
# ---------------------

for c in customers:

    if random.random() < 0.03:

        pep_id = str(uuid.uuid4())

        pep = {
            "id": pep_id,
            "list_name": random.choice(pep_list_names),
            "position": random.choice(pep_positions),
            "country": fake.country(),
            "risk": random.choice(pep_risk),
            "verified_date": fake.date_between("-3y","today")
        }

        peps.append(pep)

        cust_pep_rel.append({
            "customer_id": c["id"],
            "pep_id": pep_id
        })

# ---------------------
# Generate Alerts for customers
# ---------------------

for c in customers:

    if random.random() < 0.05:

        aid = str(uuid.uuid4())

        alert = {
            "id": aid,
            "type": random.choice(alert_types),
            "status": random.choice(alert_status),
            "severity": random.choice(alert_severity),
            "created_date": fake.date_between("-1y","today")
        }

        alerts.append(alert)

        cust_alert_rel.append({
            "customer_id": c["id"],
            "alert_id": aid
        })
# ---------------------
# Generate Alerts for Transactions
# ---------------------
for t in transactions:

    if random.random() < 0.08:

        aid = str(uuid.uuid4())

        alert = {
            "id": aid,
            "type": random.choice(alert_types),
            "status": random.choice(alert_status),
            "severity": random.choice(alert_severity),
            "created_date": fake.date_between("-1y","today")
        }

        alerts.append(alert)

        tx_alert_rel.append({
            "transaction_id": t["id"],
            "alert_id": aid
        })
# ---------------------
# Save CSVs
# ---------------------

pd.DataFrame(customers).to_csv("./data/customers.csv",index=False)
pd.DataFrame(accounts).to_csv("./data/accounts.csv",index=False)
pd.DataFrame(addresses).to_csv("./data/addresses.csv",index=False)
pd.DataFrame(transactions).to_csv("./data/transactions.csv",index=False)
pd.DataFrame(peps).to_csv("./data/peps.csv",index=False)
pd.DataFrame(alerts).to_csv("./data/alerts.csv",index=False)
pd.DataFrame(sanctions).to_csv("./data/sanctions.csv", index=False)

pd.DataFrame(cust_sanction_rel).to_csv("./data/customer_sanctions.csv", index=False)
pd.DataFrame(acct_sanction_rel).to_csv("./data/account_sanctions.csv", index=False)

pd.DataFrame(cust_pep_rel).to_csv("./data/customer_pep.csv",index=False)
pd.DataFrame(cust_alert_rel).to_csv("./data/customer_alerts.csv",index=False)
pd.DataFrame(tx_alert_rel).to_csv("./data/transaction_alerts.csv",index=False)

pd.DataFrame(cust_acc_rel).to_csv("./data/customer_accounts.csv",index=False)
pd.DataFrame(cust_addr_rel).to_csv("./data/customer_addresses.csv",index=False)

pd.DataFrame(sent_rel).to_csv("./data/transaction_sent.csv",index=False)
pd.DataFrame(recv_rel).to_csv("./data/transaction_received.csv",index=False)

print("Graph dataset generated.")