from enum import Enum

class QueryTransformationMethod(Enum):
    CONTEXTUALIZE = "contextualize"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
