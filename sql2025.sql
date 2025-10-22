
-- Schema tabella documenti con vettori di embedding 
-- TODO: da sistemare in base ai metadati richiesti effettivamente
CREATE TABLE documenti (
    id INT PRIMARY KEY,
    content NVARCHAR(MAX),
    embedding FLOAT(8) ARRAY  -- SQL Server 2025 supporta array/vettori
);

-- Funzione similarità coseno
CREATE FUNCTION cosine_similarity(@v1 FLOAT(8)[], @v2 FLOAT(8)[])
RETURNS FLOAT AS
BEGIN
    RETURN (SELECT 
                SUM(v1.elem * v2.elem) / 
                (SQRT(SUM(POWER(v1.elem, 2))) * SQRT(SUM(POWER(v2.elem, 2))))
            FROM UNNEST(@v1) AS v1(elem)
            JOIN UNNEST(@v2) AS v2(elem) ON v1.index = v2.index
           );
END;

-- Query
SELECT TOP 5 
    id,
    content,
    dbo.cosine_similarity(embedding, @query_embedding) AS similarity
FROM documenti
ORDER BY similarity DESC;