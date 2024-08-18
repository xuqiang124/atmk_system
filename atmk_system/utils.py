from typing import Union
from django.http import JsonResponse
from django.forms.models import model_to_dict


def response_success(data: Union[dict, str], msg: str = "") -> JsonResponse:
    return JsonResponse({"status": 1, "msg": msg, "data": data})


def response_error(error_msg: str = "") -> JsonResponse:
    return JsonResponse({"status": 0, "msg": error_msg})


def parse_cond(condtion) -> dict:
    """
    [
        {
            "field": "account_id",
            "eq": 1,  # equals(int)
            "lt": 1,  # less than(int)
            "gt": 1,  # greater than(int)
            "contains": "text" # contains(str)
        },
    ]
    """
    ret = {}
    for cond in condtion:
        field = cond["field"]
        if "eq" in cond:
            ret[field] = cond["eq"]
        elif "lt" in cond:
            ret[f"{field}__lt"] = cond["lt"]
        elif "gt" in cond:
            ret[f"{field}__gt"] = cond["gt"]
        elif "contains" in cond:
            ret[f"{field}__contains"] = cond["contains"]

    return ret


def collect(
    model,
    conditions=None,
    page: int = 1,
    size: int = 0,
    order_by: list = [],
    *args,
    **kwargs,
):
    count = 0
    if not conditions:
        query_sets = model.objects.all()
    else:
        query_sets = model.objects.filter(**parse_cond(conditions))
    count = len(query_sets)

    if order_by:
        query_sets = query_sets.order_by(*order_by)

    if size:
        start, end = (page - 1) * size, page * size
        query_sets = query_sets[start:end]

    ret = []
    for query in query_sets:
        temp = model_to_dict(query)
        ret.append(temp)

    return ret, count
