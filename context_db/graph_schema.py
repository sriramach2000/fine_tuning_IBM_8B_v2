"""Neo4j graph schema definitions and index creation queries."""

import os

EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# Uniqueness constraints
CONSTRAINTS = [
    "CREATE CONSTRAINT compact_context_id IF NOT EXISTS FOR (c:CompactContext) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT code_entity_id IF NOT EXISTS FOR (e:CodeEntity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
]

# Vector indexes (HNSW) for similarity search
VECTOR_INDEXES = [
    f"""CREATE VECTOR INDEX context_embedding IF NOT EXISTS
FOR (c:CompactContext) ON (c.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIMENSIONS},
  `vector.similarity_function`: 'cosine'
}}}}""",
    f"""CREATE VECTOR INDEX topic_embedding IF NOT EXISTS
FOR (t:Topic) ON (t.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIMENSIONS},
  `vector.similarity_function`: 'cosine'
}}}}""",
    f"""CREATE VECTOR INDEX code_embedding IF NOT EXISTS
FOR (e:CodeEntity) ON (e.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIMENSIONS},
  `vector.similarity_function`: 'cosine'
}}}}""",
]

# Composite indexes for fast lookups
COMPOSITE_INDEXES = [
    "CREATE INDEX context_session_ts IF NOT EXISTS FOR (c:CompactContext) ON (c.session_id, c.timestamp)",
    "CREATE INDEX code_entity_path IF NOT EXISTS FOR (e:CodeEntity) ON (e.file_path)",
    "CREATE INDEX session_project IF NOT EXISTS FOR (s:Session) ON (s.project)",
]

ALL_SCHEMA_QUERIES = CONSTRAINTS + VECTOR_INDEXES + COMPOSITE_INDEXES
