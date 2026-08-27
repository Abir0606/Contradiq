def build_filter(
    contract_type: str | None = None,
    part: str | None = None,
    clause_category: str | None = None,
) -> dict | None:
    conditions = []
    if contract_type:
        conditions.append({"contract_type": {"$in": [contract_type]}})
    if part:
        conditions.append({"part": {"$in": [part]}})
    if clause_category:
        conditions.append({"clause_categories": {"$in": [clause_category]}})
    if not conditions:
        return None
    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
