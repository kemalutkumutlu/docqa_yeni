import re

with open("app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Revert my previous multi_replace
text = text.replace("if SQLAlchemyDataLayer is not None:\n    class NativeHistoryDataLayer(SQLAlchemyDataLayer):  # type: ignore[misc]\n        _THREAD_UPSERT_COLUMNS = (\n            \"id\",", "class NativeHistoryDataLayer(SQLAlchemyDataLayer):  # type: ignore[misc]\n    _THREAD_UPSERT_COLUMNS = (\n        \"id\",")

# Fix SQLAlchemyDataLayer = None to = object
text = text.replace("SQLAlchemyDataLayer = None", "SQLAlchemyDataLayer = object")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("done")
