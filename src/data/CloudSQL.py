import os

import pg8000
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes


def connect_with_connector() -> sqlalchemy.engine.base.Engine:

    instance_connection_name = "spear-bot-388313:europe-west1:speardb"
    db_user = "postgres"
    db_pass = "L@fKJa{9~)]c`Ob%"
    db_name = "speardb"

    ip_type = IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "credentials/spear-bot-388313-6a23d6901400.json"
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # ...
    )
    return pool
