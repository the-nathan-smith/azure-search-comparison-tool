import json
import psycopg2
import pandas as pd


with open("app/backend/data/search_queries/top_searches.json", 'r') as file:
    searches = json.load(file)

resultsDConnection = psycopg2.connect(host='postgres-jiuv4g2nm7sh2.postgres.database.azure.com', database='postgres', user='resultsadmin', password='PeteIsSk1ll-')
wagtailDb = psycopg2.connect(host='nhsuk-cms-psql-dev-uks.postgres.database.azure.com', database='nhsuk-cms-review-comcomsearchpoc', user='wagtail@nhsuk-cms-psql-dev-uks', password=',Q-b-/<^)n95D99P')

resultsCursor = resultsDConnection.cursor()
wagtailCursor = wagtailDb.cursor()

for query in searches:
    i = query["query"]
    result = resultsCursor.execute(f"select * from public.poc_combined_rrf('{i}')")

    list = []
    output = []
    rank = 1
    csvFileName = f"./results/search_results_{i}.csv"
    jsonFileName = f"./results/search_results_{i}.json"

    for j in resultsCursor:
        # print("/" + j[0].replace("_", "/") + "/")
        url = "/" + j[0].replace("_", "/") + "/"
        list.append(url)
        output.append({"url": url, "score": str(j[1]), "rank": rank})
        rank += 1

    sql = f"""
        SELECT
        slug,
        title,
        url_path
        FROM public.wagtailcore_page
        WHERE url_path IN ('{"','".join(list)}')
    """

    wagtailCursor.execute(sql)
    for page in wagtailCursor:
        # print(page)
        url = page[2]
        outputresult = next((item for item in output if item.get("url") == url), None)
        outputresult["title"] = page[1]
        outputresult["slug"] = page[0]

    
        
    print(output)
    print("         ")

    # Save the JSON data to a file
    with open(jsonFileName, 'w') as file:
        json.dump(output, file)

    # Load the JSON data
    json_data = json.dumps(output)

    # Parse the JSON data into a Python dictionary
    data = pd.read_json(json_data)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv(csvFileName, index=False)

