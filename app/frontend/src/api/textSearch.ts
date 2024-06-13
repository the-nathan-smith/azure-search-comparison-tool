import axios from "axios";
import { SearchResponse, TextSearchRequest, TextSearchResult } from "./types";

export const getTextSearchResults = async (
    approach: "text" | "vec" | "vec_roshaan" | "hs" | "hssr" | undefined,
    searchQuery: string,
    useSemanticCaptions: boolean,
    dataSet?: string,
    queryVector?: number[],
    select?: string,
    k?: number
): Promise<SearchResponse<TextSearchResult>> => {
    const requestBody: TextSearchRequest = {
        query: searchQuery,
        select: select,
        dataSet: dataSet,
        approach: approach
    };

    if (approach === "vec" || approach === "vec_roshaan" || approach === "hs" || approach === "hssr") {
        requestBody.k = k;
        requestBody.queryVector = queryVector;

        if (approach === "hssr") {
            requestBody.useSemanticCaptions = useSemanticCaptions;
        }
    }

    const response = await axios.post<SearchResponse<TextSearchResult>>("/searchText", requestBody);

    return response.data;
};

export const getEmbeddings = async (query: string): Promise<number[]> => {
    const response = await axios.post<number[]>("/embedQuery", { query });
    return response.data;
};
