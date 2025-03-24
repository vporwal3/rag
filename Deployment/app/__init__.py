import json
import azure.functions as func
from .main import handle_request

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function entry point: receives an HTTP request, calls handle_request,
    and returns the result as JSON.
    """
    try:
        req_body = req.get_json()
        user_query = req_body.get("query", "")
    except ValueError:
        user_query = ""

    result = handle_request(user_query)
    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json"
    )

# import azure.functions as func
# import json

# def main(req: func.HttpRequest) -> func.HttpResponse:
#     return func.HttpResponse(
#         json.dumps({"hello": "world"}),
#         status_code=200,
#         mimetype="application/json"
#     )

