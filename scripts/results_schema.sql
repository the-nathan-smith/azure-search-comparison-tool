CREATE TABLE IF NOT EXISTS public.poc_results
(
    result_id integer GENERATED ALWAYS AS IDENTITY,
    search_query character varying(256) COLLATE pg_catalog."default" NOT NULL,
    approach_code character varying(64) COLLATE pg_catalog."default" NOT NULL,
    ndcg_3 numeric(21,20),
    ndcg_10 numeric(21,20),
    search_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT poc_results_pkey PRIMARY KEY (result_id)
)

TABLESPACE pg_default;

ALTER TABLE public.poc_results
    OWNER to resultsadmin;

CREATE TABLE IF NOT EXISTS public.poc_actual_result_rankings
(
    actual_result_ranking_id integer GENERATED ALWAYS AS IDENTITY,
    result_id integer NOT NULL,
    rank integer NOT NULL,
    article_id character varying(128) COLLATE pg_catalog."default" NOT NULL,
    relevance_score numeric(60,30),
    azure_ai_score numeric(60,30),
    CONSTRAINT poc_actual_result_rankings_pkey PRIMARY KEY (actual_result_ranking_id),
    CONSTRAINT poc_actual_result_rankings_fk_result_id FOREIGN KEY (result_id)
        REFERENCES public.poc_results (result_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE public.poc_actual_result_rankings
    OWNER to resultsadmin;

CREATE TABLE IF NOT EXISTS public.poc_ideal_result_rankings
(
    ideal_result_ranking_id integer GENERATED ALWAYS AS IDENTITY,
    result_id integer NOT NULL,
    rank integer NOT NULL,
    article_id character varying(128) COLLATE pg_catalog."default" NOT NULL,
    relevance_score numeric(60,30),
    CONSTRAINT poc_ideal_result_rankings_pkey PRIMARY KEY (ideal_result_ranking_id),
    CONSTRAINT poc_ideal_result_rankings_fk_result_id FOREIGN KEY (result_id)
        REFERENCES public.poc_results (result_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE public.poc_ideal_result_rankings
    OWNER to resultsadmin;

CREATE OR REPLACE VIEW public.poc_ndcg_rankings_3
AS
SELECT DISTINCT lower(search_query) as search_query, approach_code, ndcg_3
FROM public.poc_results
WHERE ndcg_3 IS NOT NULL
ORDER BY lower(search_query), ndcg_3 DESC, approach_code;

ALTER TABLE public.poc_ndcg_rankings_3
    OWNER TO resultsadmin;


CREATE OR REPLACE VIEW public.poc_ranked_results
AS
SELECT
    R.result_id,
    R.search_query,
    R.approach_code,
    R.ndcg_3,
    R.ndcg_10,
    ARR.rank,
    ARR.article_id,
    CAST(ARR.relevance_score AS NUMERIC(6, 3)) as relevance,
    ARR.azure_ai_score,
    IRR.article_id as expected_article_id,
    CAST(IRR.relevance_score AS NUMERIC(6, 3)) as expected_relevance
FROM public.poc_results R
    JOIN public.poc_actual_result_rankings ARR ON R.result_id = ARR.result_id
    LEFT JOIN public.poc_ideal_result_rankings IRR ON R.result_id = IRR.result_id AND IRR.rank = ARR.rank
ORDER BY R.result_id DESC, ARR.rank;

ALTER TABLE public.poc_ranked_results
    OWNER TO resultsadmin;

CREATE OR REPLACE FUNCTION public.poc_compare_search_query_results(search_query TEXT)
RETURNS TABLE(
    rank INT,
    vec_article_id TEXT,
    vec_relevance NUMERIC(6, 3),
    text_article_id TEXT,
    text_relevance NUMERIC(6, 3),
    algolia_article_id TEXT,
    algolia_relevance NUMERIC(6, 3)
)
AS $$
    WITH vec_result AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = 'vec'
        ORDER BY R.search_time DESC
        LIMIT 1),
    text_result AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = 'text'
        ORDER BY R.search_time DESC
        LIMIT 1),
    algolia_result AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = 'algolia'
        ORDER BY R.search_time DESC
        LIMIT 1)
    SELECT 
        VEC_ARR.rank,
        VEC_ARR.article_id as vec_article_id,
        CAST(VEC_ARR.relevance_score AS NUMERIC(6, 3)) AS vec_relevance,
        TEXT_ARR.article_id as text_article_id,
        CAST(TEXT_ARR.relevance_score AS NUMERIC(6, 3)) as text_relevance,
        ALG_ARR.article_id as algolia_article_id,
        CAST(ALG_ARR.relevance_score AS NUMERIC(6, 3)) as algolia_relevance
    FROM
        public.poc_actual_result_rankings VEC_ARR,
        vec_result VEC_R,
        public.poc_actual_result_rankings TEXT_ARR,
        text_result TEXT_R,
        public.poc_actual_result_rankings ALG_ARR,
        algolia_result ALG_R
    WHERE VEC_ARR.result_id = VEC_R.result_id
    AND TEXT_ARR.result_id = TEXT_R.result_id
    AND ALG_ARR.result_id = ALG_R.result_id
    AND VEC_ARR.rank = TEXT_ARR.rank
    AND VEC_ARR.rank = ALG_ARR.rank
    ORDER BY VEC_ARR.rank;
$$
LANGUAGE SQL;

ALTER FUNCTION public.poc_compare_search_query_results(search_query text)
    OWNER TO resultsadmin;

CREATE OR REPLACE FUNCTION public.poc_compare_search_query_results_2(search_query TEXT, approach_code_1 TEXT, approach_code_2 TEXT)
RETURNS TABLE(
    rank INT,
    article_id_1 TEXT,
    relevance_1 NUMERIC(6, 3),
    article_id_2 TEXT,
    relevance_2 NUMERIC(6, 3),
    algolia_article_id TEXT,
    algolia_relevance NUMERIC(6, 3)
)
AS $$
    WITH result_1 AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = $2
        ORDER BY R.search_time DESC
        LIMIT 1),
    result_2 AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = $3
        ORDER BY R.search_time DESC
        LIMIT 1),
    algolia_result AS (
        SELECT R.result_id
        FROM public.poc_results R
        WHERE lower(R.search_query) = lower($1) AND R.approach_code = 'algolia'
        ORDER BY R.search_time DESC
        LIMIT 1)
    SELECT 
        ARR_1.rank,
        ARR_1.article_id as article_id_1,
        CAST(ARR_1.relevance_score AS NUMERIC(6, 3)) AS relevance_1,
        ARR_2.article_id as article_id_2,
        CAST(ARR_2.relevance_score AS NUMERIC(6, 3)) as relevance_2,
        ALG_ARR.article_id as algolia_article_id,
        CAST(ALG_ARR.relevance_score AS NUMERIC(6, 3)) as algolia_relevance
    FROM
        public.poc_actual_result_rankings ARR_1,
        result_1 R_1,
        public.poc_actual_result_rankings ARR_2,
        result_2 R_2,
        public.poc_actual_result_rankings ALG_ARR,
        algolia_result ALG_R
    WHERE ARR_1.result_id = R_1.result_id
    AND ARR_2.result_id = R_2.result_id
    AND ALG_ARR.result_id = ALG_R.result_id
    AND ARR_1.rank = ARR_2.rank
    AND ARR_1.rank = ALG_ARR.rank
    ORDER BY ARR_1.rank;
$$
LANGUAGE SQL;

ALTER FUNCTION public.poc_compare_search_query_results_2(search_query TEXT, approach_code_1 TEXT, approach_code_2 TEXT)
    OWNER TO resultsadmin;

CREATE OR REPLACE FUNCTION public.poc_combined_rrf(search_query TEXT)
RETURNS TABLE(
    article_id_1 TEXT,
    rrf_score NUMERIC(6, 5)
)
AS $$
    WITH reciprocal_ranks AS (
        SELECT
            ARR.article_id,
            R.approach_code,
            ARR.rank + 1 as rank,
            1.0 / (ARR.rank + 1 + 60) AS reciprocal_rank
        FROM
            public.poc_actual_result_rankings ARR
        JOIN
            public.poc_results R ON ARR.result_id = R.result_id AND lower(R.search_query) = lower($1)
        WHERE R.approach_code IN ('hs_large', 'hssr_large', 'hssr_large_kw')
        ),
    aggregated_ranks AS (
        SELECT
            article_id,
            SUM(reciprocal_rank) AS rrf_score
        FROM
            reciprocal_ranks
        GROUP BY
            article_id
    )
    SELECT
        article_id,
        rrf_score
    FROM
        aggregated_ranks
    WHERE rrf_score > 0.025
    ORDER BY
        rrf_score DESC;
$$
LANGUAGE SQL;

ALTER FUNCTION public.poc_combined_rrf(search_query TEXT)
    OWNER TO resultsadmin;