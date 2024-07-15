import argparse
import base64
import os
import json
import random
import string
import time
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import uuid
import redis
import logging
import numpy as np
import requests
from keybert import KeyBERT
import torch
from transformers import AutoModel, BertTokenizer, BertModel

from openai import AzureOpenAI
#from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SemanticPrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SimpleField,
    SynonymMap,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    ExhaustiveKnnAlgorithmConfiguration,
    ScoringProfile,
    TextWeights
)
from postgres import Postgres

from azure.core.exceptions import ResourceNotFoundError

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, HealthcareEntityCategory

AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_DEPLOYMENT_LARGE_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_LARGE_NAME")
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME = os.environ.get("AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME")
AZURE_SEARCH_NHS_COMBINED_INDEX_NAME = os.environ.get("AZURE_SEARCH_NHS_COMBINED_INDEX_NAME")
AZURE_SEARCH_NHS_MSH_INDEX_NAME = os.environ.get("AZURE_SEARCH_NHS_MSH_INDEX_NAME")
AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME = os.environ.get("AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME")
AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME = os.environ.get("AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME")

REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_PASSWORD = os.environ.get("REDIS_PRIMARYKEY")

POSTGRES_SERVER_NAME = os.environ.get("POSTGRES_SERVER")
POSTGRES_SERVER_ADMIN_NAME = os.environ.get("POSTGRES_SERVER_ADMIN_LOGIN")
POSTGRES_SERVER_ADMIN_PASSWORD = os.environ.get("POSTGRES_SERVER_ADMIN_PASSWORD")

open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = "openai_token_cred"
CACHE_KEY_CREATED_TIME = "created_time"

AZURE_LANGUAGE_ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.environ.get("AZURE_LANGUAGE_KEY")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def create_and_populate_search_index_nhs_conditions():
    created = create_search_index_nhs_conditions()
    if created:
        populate_search_index_nhs_conditions()

def create_search_index_nhs_conditions():
    print(f"Ensuring search index {AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME} exists")
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    if AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME not in index_client.list_index_names():
        index = SearchIndex(
            name=AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME,
            fields=[
                SimpleField(name="id", key=True, type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SearchField(
                    name="titleVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="hnswProfile",
                ),
                SearchField(
                    name="descriptionVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="hnswProfile",
                ),
                SimpleField(
                    name="url",
                    type=SearchFieldDataType.String,
                    key=False,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                )
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="pdfHnsw")],
                profiles=[VectorSearchProfile(name="hnswProfile",
                                                algorithm_configuration_name="pdfHnsw")
                            ]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="basic-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ],
                        ),
                    )
                ]
            ),
        )
        print(f"Creating {AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME} search index")
        index_client.create_index(index)
        create_synonym_map(index_client)
        return True
    else:
        print(f"Search index {AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME} already exists")
        return False

def create_synonym_map(index_client, synonym_map_name= "default-synonym-map"):
    synonyms = [
        "diarrhoea, gastroenteritis"
    ]
    
    synonym_map = SynonymMap(name=synonym_map_name, format="solr", synonyms=synonyms)

    try:
        index_client.get_synonym_map(synonym_map_name)
        print(f"Synonym map '{synonym_map_name}' already exists.")

    except ResourceNotFoundError:

        index_client.create_synonym_map(synonym_map)
        print(f"Synonym map '{synonym_map_name}' created successfully.")

def populate_search_index_nhs_conditions():
    print(f"Populating search index {AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME} with documents")

    openai_client = AzureOpenAI(
        api_key = get_openai_key(),  
        api_version = "2024-02-01",
        azure_endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com" 
    )

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
        index_name=AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME,
    )

    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=True,
        decode_responses=True
    )

    for file_name in ["../data/conditions_1.json", "../data/medicines_1.json", "../data/articles_1.json"]:

        with open(file_name, "r", encoding="utf-8") as file:
            items = json.load(file)

        print(f"loaded {len(items)} items from {file_name}")

        batched_treated_items = []
        batch_size = 12

        for item in items:

            item_id = item["url_path"].strip('/').replace("/", "_")

            try:
                existing_doc = search_client.get_document(item_id, ["id"])

                print(f"{existing_doc["id"]} already exists. Skipping...")

                continue
            except ResourceNotFoundError:
                print(f"Adding entry for {item_id}")

            treated_item = {
                "id": item_id,
                "title": item["title"],
                "description": item["description"],
                "url": item["url_path"]
            }
            
            get_text_embeddings(
                redis_client,
                openai_client,
                text_property_names=["title", "description"],
                vector_property_names=["titleVector", "descriptionVector"],
                item=treated_item,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME)

            batched_treated_items.append(treated_item)

            if len(batched_treated_items) >= batch_size:

                print(f"Uploading batch of {len(batched_treated_items)} items ...")

                search_client.upload_documents(batched_treated_items)

                batched_treated_items.clear()

        if len(batched_treated_items) > 0:

            print(f"Uploading final batch of {len(batched_treated_items)} items ...")
            search_client.upload_documents(batched_treated_items)

        print(f"Uploaded {len(items)} documents to index '{AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME}'")

def delete_search_index(name: str):
    print(f"Deleting search index {name}")
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    index_client.delete_index(name)

def clean_text(raw_text: str) -> str:

    # nbsp = u'\xa0'
    unicode_nbsp = "\\u00a0"
    
    soup = BeautifulSoup(raw_text.replace(unicode_nbsp, " ").encode().decode('unicode-escape'), 'html.parser')

    return soup.get_text(separator=" ").replace("  "," ")

def generate_vectors(text):

    client = AzureOpenAI(
        api_key = get_openai_key(),  
        api_version = "2024-02-01",
        azure_endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com" 
    )

    response = client.embeddings.create(
        input = text,
        model = AZURE_OPENAI_DEPLOYMENT_NAME  # model = "deployment_name".
    )

    return response.data[0].embedding

def generate_text_embeddings(
        client: AzureOpenAI,
        text: any,
        openai_deployment_name = AZURE_OPENAI_DEPLOYMENT_NAME):
    
    response = client.embeddings.create(
        input = text,
        model = openai_deployment_name)

    embeddings = np.array([item.embedding for item in response.data])

    return np.concatenate(embeddings).tolist()
    
def get_text_embeddings(
        redis_client,
        openai_client,
        text_property_names: list,
        vector_property_names: list,
        item,
        deployment_name: str):

    if (len(text_property_names) != len(vector_property_names)):
        raise Exception("The number of text property names must equal the number of vector property names") 

    item_key = f"{item["id"]}_{deployment_name}"

    if redis_client.exists(item_key):
        print(f"Document with id {item["id"]} already exists in Redis cache, retrieving vectors.")

        cached_item = json.loads(redis_client.get(item_key))

        for vector_property_name in vector_property_names:
            if vector_property_name in cached_item:
                item[vector_property_name] = cached_item[vector_property_name]
    else:
        print(f"Generating Azure OpenAI embeddings for {item["id"]} ...")

        redis_obj = {}

        for idx, text_property_name in enumerate(text_property_names):

            if text_property_name in item:

                text = item[text_property_name] if item[text_property_name] is str else " ".join(item[text_property_name])

                embeddings = generate_text_embeddings(openai_client, text, deployment_name)

                vector_property_name = vector_property_names[idx]

                item[vector_property_name] = embeddings

                redis_obj[vector_property_name] = embeddings

        # Store vectors in Redis
        redis_client.set(item_key, json.dumps(redis_obj))

def get_text_embeddings_medbert(
        text_property_names: list,
        vector_property_names: list,
        item,
        vector_model,
        tokenizer
        ):

    if (len(text_property_names) != len(vector_property_names)):
        raise Exception("The number of text property names must equal the number of vector property names") 

    print(f"Generating MedBert embeddings for {item["id"]} ...")

    for idx, text_property_name in enumerate(text_property_names):

        if text_property_name in item:

            embeddings = generate_medbert_vector_embeddings(chunk_text(item[text_property_name], tokenizer), tokenizer, vector_model)

            vector_property_name = vector_property_names[idx]

            item[vector_property_name] = embeddings

def get_text_embeddings_bert(
        text_property_names: list,
        vector_property_names: list,
        item,
        vector_model,
        tokenizer
        ):

    if (len(text_property_names) != len(vector_property_names)):
        raise Exception("The number of text property names must equal the number of vector property names") 

    print(f"Generating BERT embeddings for {item["id"]} ...")

    for idx, text_property_name in enumerate(text_property_names):

        if text_property_name in item:

            embeddings = generate_bert_vector_embeddings(item[text_property_name], tokenizer, vector_model)

            vector_property_name = vector_property_names[idx]

            item[vector_property_name] = embeddings

def get_openai_key():

    if (not CACHE_KEY_CREATED_TIME in open_ai_token_cache) or open_ai_token_cache[CACHE_KEY_CREATED_TIME] + 300 < time.time():

        openai_token = azure_credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )

        open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
        open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = openai_token
    else:
        openai_token = open_ai_token_cache[CACHE_KEY_TOKEN_CRED]

    return openai_token.token

def publish_results_db_schema():

    print('Ensuring postgres results db schema exists')

    try:
        db = Postgres(f"postgresql://{POSTGRES_SERVER_ADMIN_NAME}:{quote_plus(POSTGRES_SERVER_ADMIN_PASSWORD)}@{POSTGRES_SERVER_NAME}.postgres.database.azure.com:5432/postgres?sslmode=require")

        sql_file = open('./scripts/results_schema.sql','r')

        with db.get_cursor() as cursor:
            cursor.execute(sql_file.read())

    except Exception as e:
        logging.exception(str(e))

def create_search_index_nhs_combined_data() -> bool:
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    create_synonym_map(index_client, synonym_map_name="test-sm")

    if AZURE_SEARCH_NHS_COMBINED_INDEX_NAME not in index_client.list_index_names():
        index = SearchIndex(
            name=AZURE_SEARCH_NHS_COMBINED_INDEX_NAME,
            fields=[
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                    facetable=True,
                ),
                SearchableField(name="title", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="description", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="aspect_headers", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="short_descriptions", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="content", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchField(
                    name="title_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="aspect_headers_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="short_descriptions_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=False
                ),
                SearchField(
                    name="content_types",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=False,
                    filterable=True,
                    sortable=False,
                    facetable=True
                ),
                SimpleField(
                    name="url",
                    type=SearchFieldDataType.String,
                    key=False,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                )
            ],
            vector_search=VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="exhaustiveKnn")],
                profiles=[VectorSearchProfile(
                    name="knn-vector-profile",
                    algorithm_configuration_name="exhaustiveKnn")]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-content",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description"),
                                SemanticField(field_name="short_descriptions")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-keywords",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="keywords")
                            ]
                        ),
                    )
                ]
            ),
            scoring_profiles=[
                ScoringProfile(
                    name="title_weighted",
                    text_weights=TextWeights(
                        weights={
                            "title": 2.0,
                            "description": 1.75,
                            "aspect_headers": 1.5,
                            "short_descriptions": 1.25,
                            })),
                ScoringProfile(
                    name="title_weighted_100",
                    text_weights=TextWeights(
                        weights={
                            "title": 100.0,
                            "description": 1.75,
                            "aspect_headers": 1.5,
                            "short_descriptions": 1.25,
                            }))
            ]
        )
        print(f"Creating {AZURE_SEARCH_NHS_COMBINED_INDEX_NAME} search index")
        index_client.create_index(index)
        return True
    else:
        print(f"Search index {AZURE_SEARCH_NHS_COMBINED_INDEX_NAME} already exists")
        return True

def populate_search_index_nhs_combined_data():
    print(f"Populating search index {AZURE_SEARCH_NHS_COMBINED_INDEX_NAME} with documents")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
        index_name=AZURE_SEARCH_NHS_COMBINED_INDEX_NAME,
    )

    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=True,
        decode_responses=True
    )

    openai_client = AzureOpenAI(
        api_key = get_openai_key(),  
        api_version = "2024-02-01",
        azure_endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com" 
    )

    kw_model = KeyBERT()

    for file_name in ["data/conditions_1.json", "data/medicines_1.json", "data/articles_1.json"]:

        with open(file_name, "r", encoding="utf-8") as file:
            items = json.load(file)

        print(f"loaded {len(items)} items from {file_name}")

        batched_treated_items = []
        batch_size = 12

        for item in items:

            item_id = item["url_path"].strip('/').replace("/", "_")

            try:
                existing_doc = search_client.get_document(item_id, ["id"])

                print(f"{existing_doc["id"]} already exists. Skipping...")

                continue
            except ResourceNotFoundError:
                print(f"Adding entry for {item_id}")

            treated_item = {
                "id": item_id,
                "title": item["title"],
                "description": item["description"],
                "content_types": item["content_types"],
                "url": item["url_path"]
            }

            aspect_headers = []
            short_descriptions = []
            rich_text_content = []

            all_content = ""

            for c in item["content"]:

                aspect_header = clean_text(c["value"]["aspect_header"])

                if len(aspect_header) > 0:
                    aspect_headers.append(aspect_header)

                short_description = c["value"]["short_description"]

                if len(short_description) > 0:
                    short_descriptions.append(short_description)

                for inner_c in c["value"]["content"]:

                    if inner_c["type"] == "richtext":

                        content = clean_text(inner_c["value"])

                        rich_text_content.append(content)

                        if len(all_content) == 0:
                            all_content = content
                        else:
                            all_content += " " + content

            print(f"{treated_item["id"]} has {len(aspect_headers)} aspect headers and {len(short_descriptions)} short descriptions")

            if len(aspect_headers) > 0:
                treated_item["aspect_headers"] = aspect_headers

            if len(short_descriptions) > 0:
                treated_item["short_descriptions"] = short_descriptions

            if len(rich_text_content) > 0:
                treated_item["content"] = rich_text_content
            
            get_text_embeddings(
                redis_client,
                openai_client,
                text_property_names=["title", "description", "aspect_headers", "short_descriptions", "content"],
                vector_property_names=["title_vector", "description_vector", "aspect_headers_vector", "short_descriptions_vector", "content_vector"],
                item=treated_item,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_LARGE_NAME)

            treated_item["keywords"] = get_keywords(kw_model, treated_item)

            batched_treated_items.append(treated_item)

            if len(batched_treated_items) >= batch_size:

                print(f"Uploading batch of {len(batched_treated_items)} items ...")

                search_client.upload_documents(batched_treated_items)

                batched_treated_items.clear()

        if len(batched_treated_items) > 0:

            print(f"Uploading final batch of {len(batched_treated_items)} items ...")
            search_client.upload_documents(batched_treated_items)

        print(f"Uploaded {len(items)} documents to index '{AZURE_SEARCH_NHS_COMBINED_INDEX_NAME}'")

def get_keywords(kw_model, item) -> list[str]:
    str_aspect_headers = " ".join(item["aspect_headers"]) if "aspect_headers" in item else ""
    str_short_descs = " ".join(item["short_descriptions"]) if "short_descriptions" in item else ""
    str_content = " ".join(item["content"]) if "content" in item else ""

    doc = " ".join([ item["title"], item["description"], str_aspect_headers, str_short_descs, str_content ])

    keywords = kw_model.extract_keywords(
        doc,
        top_n=20,
        use_mmr=True,
        diversity=0.5)

    selected_keywords = []

    for kw in keywords:
        if kw[1] > 0.25:
            selected_keywords.append(kw[0])

    # print(selected_keywords)

    return selected_keywords

def get_mesh_entity_data(entity_id: str):

    try:
        with open(f"./results/MESH/{entity_id}.json", 'rt') as f:
            return json.load(f)
    except OSError:
        print(f"No existing file for {entity_id}")

    try:
        response = requests.post(f"https://id.nlm.nih.gov/mesh/{entity_id}.json")

        if response.status_code == 200:
            print(f"successful MESH query for: {entity_id}")

            json_body = response.json()

            with open(f"./results/MESH/{entity_id}.json", 'w') as f:
                json.dump(json_body, f, indent=2)

            return json_body
    except:
        print(f"MESH query failed for: {entity_id}")

    return None

def get_related_mesh_data(entity_id: str, medicine_name_keywords: list[str], processed_mesh_entity_ids: list[str]):
    if entity_id in processed_mesh_entity_ids:
        print(f"Already processed MESH entity {entity_id}. stopping this branch")
        return

    json_body = get_mesh_entity_data(entity_id)

    if json_body is None:
        print(f"No data for MESH entity {entity_id}. stopping this branch")
        return

    context = json_body["@context"]

    if "label" in context:

        label = json_body["label"]["@value"]

        if label not in medicine_name_keywords:
            medicine_name_keywords.append(label)

        if "preferredTerm" in context:
            preferred_term_entity_id = json_body["preferredTerm"].split('/')[-1]

            # print(f"Preferred Term: {json_body["preferredTerm"]}")

            get_related_mesh_data(preferred_term_entity_id, medicine_name_keywords, processed_mesh_entity_ids)

        if "concept" in context:

            if isinstance(json_body["concept"], list):

                for concept in json_body["concept"]:

                    # print(f"Concept: {concept}")

                    concept_entity_id = concept.split('/')[-1]

                    get_related_mesh_data(concept_entity_id, medicine_name_keywords, processed_mesh_entity_ids)
            else:

                # print(f"Concept: {json_body["concept"]}")

                concept_entity_id = json_body["concept"].split('/')[-1]

                get_related_mesh_data(concept_entity_id, medicine_name_keywords, processed_mesh_entity_ids)


    if "prefLabel" in context:

        preferred_label = json_body["prefLabel"]["@value"]

        if preferred_label not in medicine_name_keywords:
            medicine_name_keywords.append(preferred_label)

    processed_mesh_entity_ids.append(entity_id)

def get_medicine_name_keywords(text_analytics_client: TextAnalyticsClient, item) -> list[str]:

    documents = [
        item["title"],
        item["description"]
    ]

    if "aspect_headers" in item:
        documents = documents + item["aspect_headers"]

    if "short_descriptions" in item:
        documents = documents + item["short_descriptions"]

    if "content" in item:
        documents = documents + item["content"]

    docs = []

    for batch in [documents[i:i+20] for i in range(0,len(documents),20)]:
        # print(batch)

        poller = text_analytics_client.begin_analyze_healthcare_entities(batch, language="en")
        result = poller.result()

        docs = docs + [doc for doc in result if not doc.is_error]

    mesh_entity_ids = []
    medicine_name_keywords = []

    for doc in docs:

        entities = [entity for entity in doc.entities if entity.category == HealthcareEntityCategory.MEDICATION_NAME]

        for entity in entities:

            if entity.confidence_score < 0.8:
                print(f"{entity.text} has low confidence: {entity.confidence_score}. Skipping...")
                break

            if entity.data_sources is not None:

                mesh_data_sources = [data_source for data_source in entity.data_sources if data_source.name == "MSH"]

                if len(mesh_data_sources) > 0:

                    print(f"Acquiring MESH Data Source info for medication {entity.text}:")

                    for data_source in mesh_data_sources:

                        get_related_mesh_data(data_source.entity_id, medicine_name_keywords, mesh_entity_ids)

    print("MEDICINE NAME KEYWORDS:")
    print(medicine_name_keywords)

    return sorted(medicine_name_keywords)

def create_and_populate_nhs_combined_data_index():
    created = create_search_index_nhs_combined_data()
    if created:
        populate_search_index_nhs_combined_data()

def chunk_text(text: str | list[str], tokenizer: BertTokenizer, max_length = 510) -> list[str]:

    chunked_texts = []

    if isinstance(text, list):

        for t in text:
            tokens = tokenizer.tokenize(t)

            chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        
            # Detokenize chunks back into strings
            chunked_texts += [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
    else:
        # Tokenize the input text
        tokens = tokenizer.tokenize(text)
        
        # Split the tokens into chunks
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        
        # Detokenize chunks back into strings
        chunked_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
    
    return chunked_texts

def generate_medbert_vector_embeddings(text: list[str], tokenizer: BertTokenizer, model: BertModel):

    if len(text) == 0:
        return []

    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # if encoded_input["input_ids"].shape[1] > 512:
    #     print(f"TRUNCATING text. {encoded_input["input_ids"].shape}")
    #     truncation_counter += 1

            # print(f"TOKEN TYPE IDS - {encoded_input["token_type_ids"].shape}")
            # print(f"ATTENTION MASK - {encoded_input["attention_mask"].shape}")

            # for ids in encoded_input["input_ids"]:
            #     print(ids)
            #     marked_text = tokenizer.decode(ids)
            #     print(tokenizer.decode(ids))

            #     print(tokenizer.tokenize(marked_text))

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state

    # Using the [CLS] token embeddings for each input in the batch
    cls_embeddings = hidden_states[:, 0, :]  # Extract the [CLS] token representations
    # print("CLS Embeddings Shape:", cls_embeddings.shape)  # Shape: (batch_size, hidden_size)

    # Mean pooling: Compute the mean of all [CLS] embeddings
    mean_pooled_embedding = torch.mean(cls_embeddings, dim=0)
    # print("Mean Pooled Embedding Shape:", mean_pooled_embedding.shape)  # Shape: (hidden_size,)

    # Max pooling
    max_pooled_embedding = torch.max(cls_embeddings, dim=0).values
    # print("Max Pooled Embedding Shape:", max_pooled_embedding.shape)  # Shape: (hidden_size,)

    # Min pooling
    min_pooled_embedding = torch.min(cls_embeddings, dim=0).values
    # print("Min Pooled Embedding Shape:", min_pooled_embedding.shape)  # Shape: (hidden_size,)

    # Concatenate mean, max, and min pooled embeddings
    concatenated_embedding = torch.cat((mean_pooled_embedding, max_pooled_embedding, min_pooled_embedding), dim=0)
    # print("Concatenated Embedding Shape:", concatenated_embedding.shape)  # Shape: (3 * hidden_size,)

    return concatenated_embedding.tolist()

def generate_bert_vector_embeddings(text: str, tokenizer: BertTokenizer, model: BertModel):

    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state

    # Using the [CLS] token embeddings for each input in the batch
    cls_embeddings = hidden_states[:, 0, :]  # Extract the [CLS] token representations
    print("CLS Embeddings Shape:", cls_embeddings.shape)  # Shape: (batch_size, hidden_size)

    return cls_embeddings.squeeze().tolist()

def create_search_index_nhs_medbert_data() -> bool:
    print(type(AZURE_SEARCH_SERVICE_ENDPOINT))
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    create_synonym_map(index_client, synonym_map_name="test-sm")

    if AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME not in index_client.list_index_names():
        index = SearchIndex(
            name=AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME,
            fields=[
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                    facetable=True,
                ),
                SearchableField(name="title", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="description", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="aspect_headers", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="short_descriptions", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="content", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchField(
                    name="title_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=2304,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=2304,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="aspect_headers_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=2304,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="short_descriptions_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=2304,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=2304,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True
                ),
                SearchField(
                    name="content_types",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=False,
                    filterable=True,
                    sortable=False,
                    facetable=True
                ),
                SimpleField(
                    name="url",
                    type=SearchFieldDataType.String,
                    key=False,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                )
            ],
            vector_search=VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="exhaustiveKnn")],
                profiles=[VectorSearchProfile(
                    name="knn-vector-profile",
                    algorithm_configuration_name="exhaustiveKnn")]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-content",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description"),
                                SemanticField(field_name="short_descriptions")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-keywords",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="keywords")
                            ]
                        ),
                    )
                ]
            ),
            scoring_profiles=[
                ScoringProfile(
                    name="title_weighted",
                    text_weights=TextWeights(
                        weights={
                            "title": 10.0,
                            "keywords": 5.0,
                            "description": 2.0,
                            "aspect_headers": 1.5,
                            "short_descriptions": 1.25,
                            "content": 1.0
                            }))
            ]
        )
        print(f"Creating {AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME} search index")
        index_client.create_index(index)
        return True
    else:
        print(f"Search index {AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME} already exists")
        return True

def populate_search_index_nhs_medbert_data():
    print(f"Populating search index {AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME} with documents")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
        index_name=AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME,
    )

    model_name = "Charangan/MedBERT"
    model = AutoModel.from_pretrained(model_name)
    kw_model = KeyBERT(model=model)

    # Set a random seed
    random_seed = 42
    random.seed(random_seed)
    
    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    vector_model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    vector_model.eval()

    processed_item_ids = []

    try:
        with open("medbert_item_ids.json", encoding="utf-8") as f:
            processed_item_ids = [line.rstrip() for line in f]

        print(processed_item_ids)

    except:
        print("processed item ids file not found")

    for file_name in ["data/conditions_1.json", "data/medicines_1.json", "data/articles_1.json"]:

        with open(file_name, "r", encoding="utf-8") as file:
            items = json.load(file)

        print(f"loaded {len(items)} items from {file_name}")

        batched_treated_items = []
        batch_size = 12

        for item in items:

            item_id = item["url_path"].strip('/').replace("/", "_").replace("â","a").replace("ŷ","y")

            if item_id in processed_item_ids:
                print(f"{item_id} already exists. Skipping...")
                continue

            try:
                existing_doc = search_client.get_document(item_id, ["id"])

                print(f"{existing_doc["id"]} already exists. Skipping...")

                with open("medbert_item_ids.json", 'a') as f: 
                    f.write(item_id + '\n') 

                continue
            except ResourceNotFoundError:
                print(f"Adding entry for {item_id}")

            treated_item = {
                "id": item_id,
                "title": item["title"],
                "description": item["description"],
                "content_types": item["content_types"],
                "url": item["url_path"]
            }

            aspect_headers = []
            short_descriptions = []
            rich_text_content = []

            all_content = ""

            for c in item["content"]:

                aspect_header = clean_text(c["value"]["aspect_header"])

                if len(aspect_header) > 0:
                    aspect_headers.append(aspect_header)

                short_description = c["value"]["short_description"]

                if len(short_description) > 0:
                    short_descriptions.append(short_description)

                for inner_c in c["value"]["content"]:

                    if inner_c["type"] == "richtext":

                        content = clean_text(inner_c["value"])

                        rich_text_content.append(content)

                        if len(all_content) == 0:
                            all_content = content
                        else:
                            all_content += " " + content

            if len(aspect_headers) > 0:
                treated_item["aspect_headers"] = aspect_headers

            if len(short_descriptions) > 0:
                treated_item["short_descriptions"] = short_descriptions

            if len(rich_text_content) > 0:
                treated_item["content"] = rich_text_content

            treated_item["keywords"] = get_keywords(kw_model, treated_item)

            get_text_embeddings_medbert(
                text_property_names=["title", "description", "aspect_headers", "short_descriptions", "content"],
                vector_property_names=["title_vector", "description_vector", "aspect_headers_vector", "short_descriptions_vector", "content_vector"],
                item=treated_item,
                vector_model=vector_model,
                tokenizer=tokenizer
            )

            batched_treated_items.append(treated_item)

            print(f"Successfully generated embedding for: {treated_item['title']}")

            if len(batched_treated_items) >= batch_size:

                print(f"Uploading batch of {len(batched_treated_items)} items ...")

                search_client.upload_documents(batched_treated_items)

                batched_treated_items.clear()
        
        if len(batched_treated_items) > 0:

            print(f"Uploading final batch of {len(batched_treated_items)} items ...")
            search_client.upload_documents(batched_treated_items)

        print(f"Uploaded {len(items)} documents to index '{AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME}'")

def create_and_populate_nhs_medbert_data_index():
    created = create_search_index_nhs_medbert_data()
    if created:
        populate_search_index_nhs_medbert_data()

def create_search_index_nhs_msh() -> bool:
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    create_synonym_map(index_client, synonym_map_name="test-sm")

    if AZURE_SEARCH_NHS_MSH_INDEX_NAME not in index_client.list_index_names():
        index = SearchIndex(
            name=AZURE_SEARCH_NHS_MSH_INDEX_NAME,
            fields=[
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                    facetable=True,
                ),
                SearchableField(name="title", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="description", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="aspect_headers", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="short_descriptions", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchableField(name="content", collection=True, type=SearchFieldDataType.Collection(SearchFieldDataType.String), synonym_map_names=["test-sm"]),
                SearchField(
                    name="title_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="aspect_headers_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="short_descriptions_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True
                ),
                SearchField(
                    name="medicine_name_keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True
                ),
                SearchField(
                    name="content_types",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=False,
                    filterable=True,
                    sortable=False,
                    facetable=True
                ),
                SimpleField(
                    name="url",
                    type=SearchFieldDataType.String,
                    key=False,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                )
            ],
            vector_search=VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="exhaustiveKnn")],
                profiles=[VectorSearchProfile(
                    name="knn-vector-profile",
                    algorithm_configuration_name="exhaustiveKnn")]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="semantic-config-keywords",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description"),
                                SemanticField(field_name="short_descriptions")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="medicine_name_keywords"),
                                SemanticField(field_name="keywords")
                            ]
                        ),
                    )
                ]
            ),
            scoring_profiles=[
                ScoringProfile(
                    name="title_weighted",
                    text_weights=TextWeights(
                        weights={
                            "title": 10.0,
                            "keywords": 5.0,
                            "description": 2.0,
                            "aspect_headers": 1.5,
                            "short_descriptions": 1.25,
                            "content": 1.0,
                            "medicine_name_keywords": 0.75,
                            }))
            ]
        )
        print(f"Creating {AZURE_SEARCH_NHS_MSH_INDEX_NAME} search index")
        index_client.create_index(index)
        return True
    else:
        print(f"Search index {AZURE_SEARCH_NHS_MSH_INDEX_NAME} already exists")
        return True

def populate_search_index_nhs_msh():
    print(f"Populating search index {AZURE_SEARCH_NHS_MSH_INDEX_NAME} with documents")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
        index_name=AZURE_SEARCH_NHS_MSH_INDEX_NAME,
    )

    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=True,
        decode_responses=True
    )

    openai_client = AzureOpenAI(
        api_key = get_openai_key(),  
        api_version = "2024-02-01",
        azure_endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com" 
    )

    kw_model = KeyBERT()

    text_analytics_client = TextAnalyticsClient(
        endpoint=AZURE_LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_LANGUAGE_KEY),
    )

    processed_item_ids = []

    try:
        with open("item_ids.json", encoding="utf-8") as f:
            processed_item_ids = [line.rstrip() for line in f]

        print(processed_item_ids)

    except:
        print("processed item ids file not found")

    for file_name in ["data/conditions_1.json", "data/medicines_1.json", "data/articles_1.json"]:

        with open(file_name, "r", encoding="utf-8") as file:
            items = json.load(file)

        print(f"loaded {len(items)} items from {file_name}")

        batched_treated_items = []
        batch_size = 12

        for item in items:

            item_id = item["url_path"].strip('/').replace("/", "_").replace("â","a").replace("ŷ","y")

            if item_id in processed_item_ids:
                print(f"{item_id} already exists. Skipping...")
                continue

            try:
                existing_doc = search_client.get_document(item_id, ["id"])

                print(f"{existing_doc["id"]} already exists. Skipping...")

                with open("item_ids.json", 'a') as f: 
                    f.write(item_id + '\n') 

                continue
            except ResourceNotFoundError:
                print(f"Adding entry for {item_id}")

            treated_item = {
                "id": item_id,
                "title": item["title"],
                "description": item["description"],
                "content_types": item["content_types"],
                "url": item["url_path"]
            }

            aspect_headers = []
            short_descriptions = []
            rich_text_content = []

            all_content = ""

            for c in item["content"]:

                aspect_header = clean_text(c["value"]["aspect_header"])

                if len(aspect_header) > 0:
                    aspect_headers.append(aspect_header)

                short_description = c["value"]["short_description"]

                if len(short_description) > 0:
                    short_descriptions.append(short_description)

                for inner_c in c["value"]["content"]:

                    if inner_c["type"] == "richtext":

                        content = clean_text(inner_c["value"])

                        rich_text_content.append(content)

                        if len(all_content) == 0:
                            all_content = content
                        else:
                            all_content += " " + content

            print(f"{treated_item["id"]} has {len(aspect_headers)} aspect headers and {len(short_descriptions)} short descriptions")

            if len(aspect_headers) > 0:
                treated_item["aspect_headers"] = aspect_headers

            if len(short_descriptions) > 0:
                treated_item["short_descriptions"] = short_descriptions

            if len(rich_text_content) > 0:
                treated_item["content"] = rich_text_content
            
            get_text_embeddings(
                redis_client,
                openai_client,
                text_property_names=["title", "description", "aspect_headers", "short_descriptions", "content"],
                vector_property_names=["title_vector", "description_vector", "aspect_headers_vector", "short_descriptions_vector", "content_vector"],
                item=treated_item,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_LARGE_NAME)

            treated_item["keywords"] = get_keywords(kw_model, treated_item)
            treated_item["medicine_name_keywords"] = get_medicine_name_keywords(text_analytics_client, treated_item)

            batched_treated_items.append(treated_item)

            if len(batched_treated_items) >= batch_size:

                print(f"Uploading batch of {len(batched_treated_items)} items ...")

                search_client.upload_documents(batched_treated_items)

                batched_treated_items.clear()

        if len(batched_treated_items) > 0:

            print(f"Uploading final batch of {len(batched_treated_items)} items ...")
            search_client.upload_documents(batched_treated_items)

        print(f"Uploaded {len(items)} documents to index '{AZURE_SEARCH_NHS_MSH_INDEX_NAME}'")

def create_and_populate_nhs_msh_index():
    created = create_search_index_nhs_msh()
    if created:
        populate_search_index_nhs_msh()

def create_search_index_nhs_bert_base() -> bool:
    print(type(AZURE_SEARCH_SERVICE_ENDPOINT))
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
    )
    create_synonym_map(index_client, synonym_map_name="test-sm")

    if AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME not in index_client.list_index_names():
        index = SearchIndex(
            name=AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME,
            fields=[
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                    facetable=True,
                ),
                SearchableField(name="title", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchableField(name="description", type=SearchFieldDataType.String, synonym_map_names=["test-sm"]),
                SearchField(
                    name="title_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=768,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="description_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=768,
                    vector_search_profile_name="knn-vector-profile"
                ),
                SearchField(
                    name="keywords",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True
                ),
                SearchField(
                    name="content_types",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=False,
                    filterable=True,
                    sortable=False,
                    facetable=True
                ),
                SimpleField(
                    name="url",
                    type=SearchFieldDataType.String,
                    key=False,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                )
            ],
            vector_search=VectorSearch(
                algorithms=[ExhaustiveKnnAlgorithmConfiguration(name="exhaustiveKnn")],
                profiles=[VectorSearchProfile(
                    name="knn-vector-profile",
                    algorithm_configuration_name="exhaustiveKnn")]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-content",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ]
                        ),
                    ),
                    SemanticConfiguration(
                        name="semantic-config-keywords",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[
                                SemanticField(field_name="description")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="keywords")
                            ]
                        ),
                    )
                ]
            ),
            scoring_profiles=[
                ScoringProfile(
                    name="title_weighted",
                    text_weights=TextWeights(
                        weights={
                            "title": 5.0,
                            "keywords": 2.0,
                            "description": 1.0
                            }))
            ]
        )
        print(f"Creating {AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME} search index")
        index_client.create_index(index)
        return True
    else:
        print(f"Search index {AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME} already exists")
        return True

def populate_search_index_nhs_bert_base():
    print(f"Populating search index {AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME} with documents")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=azure_credential,
        index_name=AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME,
    )

    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    kw_model = KeyBERT(model=model)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    vector_model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    vector_model.eval()

    processed_item_ids = []

    try:
        with open("bert_base_item_ids.json", encoding="utf-8") as f:
            processed_item_ids = [line.rstrip() for line in f]

        print(processed_item_ids)

    except:
        print("processed item ids file not found")

    for file_name in ["data/conditions_1.json", "data/medicines_1.json", "data/articles_1.json"]:

        with open(file_name, "r", encoding="utf-8") as file:
            items = json.load(file)

        print(f"loaded {len(items)} items from {file_name}")

        batched_treated_items = []
        batch_size = 12

        for item in items:

            item_id = item["url_path"].strip('/').replace("/", "_").replace("â","a").replace("ŷ","y")

            if item_id in processed_item_ids:
                print(f"{item_id} already exists. Skipping...")
                continue

            try:
                existing_doc = search_client.get_document(item_id, ["id"])

                print(f"{existing_doc["id"]} already exists. Skipping...")

                with open("medbert_item_ids.json", 'a') as f: 
                    f.write(item_id + '\n') 

                continue
            except ResourceNotFoundError:
                print(f"Adding entry for {item_id}")

            treated_item = {
                "id": item_id,
                "title": item["title"],
                "description": item["description"],
                "content_types": item["content_types"],
                "url": item["url_path"]
            }

            aspect_headers = []
            short_descriptions = []
            rich_text_content = []

            all_content = ""

            for c in item["content"]:

                aspect_header = clean_text(c["value"]["aspect_header"])

                if len(aspect_header) > 0:
                    aspect_headers.append(aspect_header)

                short_description = c["value"]["short_description"]

                if len(short_description) > 0:
                    short_descriptions.append(short_description)

                for inner_c in c["value"]["content"]:

                    if inner_c["type"] == "richtext":

                        content = clean_text(inner_c["value"])

                        rich_text_content.append(content)

                        if len(all_content) == 0:
                            all_content = content
                        else:
                            all_content += " " + content

            if len(aspect_headers) > 0:
                treated_item["aspect_headers"] = aspect_headers

            if len(short_descriptions) > 0:
                treated_item["short_descriptions"] = short_descriptions

            if len(rich_text_content) > 0:
                treated_item["content"] = rich_text_content

            treated_item["keywords"] = get_keywords(kw_model, treated_item)

            if "aspect_headers" in treated_item:
                del treated_item["aspect_headers"]

            if "short_descriptions" in treated_item:
                del treated_item["short_descriptions"]

            if "content" in treated_item:
                del treated_item["content"]

            get_text_embeddings_bert(
                text_property_names=["title", "description"],
                vector_property_names=["title_vector", "description_vector"],
                item=treated_item,
                vector_model=vector_model,
                tokenizer=tokenizer
            )

            batched_treated_items.append(treated_item)

            print(f"Successfully generated embedding for: {treated_item['title']}")

            if len(batched_treated_items) >= batch_size:

                print(f"Uploading batch of {len(batched_treated_items)} items ...")

                search_client.upload_documents(batched_treated_items)

                batched_treated_items.clear()
        
        if len(batched_treated_items) > 0:

            print(f"Uploading final batch of {len(batched_treated_items)} items ...")
            search_client.upload_documents(batched_treated_items)

        print(f"Uploaded {len(items)} documents to index '{AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME}'")


def create_and_populate_nhs_bert_base_index():
    created = create_search_index_nhs_bert_base()
    if created:
        populate_search_index_nhs_bert_base()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepares the required Azure Cognitive Search indexes for the app",
    )
    parser.add_argument(
        "--recreate",
        required=False,
        action="store_true",
        help="Optional. Recreate all the ACS indexes",
    )
    args = parser.parse_args()

    # Use the current user identity to connect to Azure services
    azure_credential = DefaultAzureCredential(
        exclude_shared_token_cache_credential=True
    )

    # Create result db schema
    publish_results_db_schema()

    # Create NHS bert index
    if args.recreate:
        delete_search_index(AZURE_SEARCH_NHS_BERT_BASE_INDEX_NAME)
    create_and_populate_nhs_bert_base_index()

    # Create NHS combined data set index
    if args.recreate:
      delete_search_index(AZURE_SEARCH_NHS_COMBINED_INDEX_NAME)
    create_and_populate_nhs_combined_data_index()

    # Create NHS conditions index
    if args.recreate:
        delete_search_index(AZURE_SEARCH_NHS_CONDITIONS_INDEX_NAME)
    create_and_populate_search_index_nhs_conditions()

    # Create NHS medbert index
    if args.recreate:
        delete_search_index(AZURE_SEARCH_NHS_MEDBERT_INDEX_NAME)
    create_and_populate_nhs_medbert_data_index()

    # Create NHS MSH index
    if args.recreate:
        delete_search_index(AZURE_SEARCH_NHS_MSH_INDEX_NAME)
    create_and_populate_nhs_msh_index()
 
    print("Completed successfully")
