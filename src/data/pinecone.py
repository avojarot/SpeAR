import uuid

import pinecone

INDEX_NAME = "speakers"
IMB_DIM = 512


def create_index(name=INDEX_NAME, dim=IMB_DIM):
    # pinecone.create_index(name=name, dimension=dim, metric='cosine')
    index = pinecone.Index(name)
    return index


def insert_row(index, vector, user_id, name):
    metadata = {"user": user_id, "name": name}
    index.upsert(
        vectors=[{"id": str(uuid.uuid4()), "values": vector, "metadata": metadata}],
    )
    return name


# TODO add to predict
def query_voices():
    pass
