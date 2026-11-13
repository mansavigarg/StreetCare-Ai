import pytest


@pytest.fixture(scope="session")
def processed_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.CLEAN_REPORT_HISTORY_TRAIN_FILE,
                          config.CLEAN_REPORT_HISTORY_EVAL_FILE,
                          config.CLEAN_REPAIR_LOGS_TRAIN_FILE,
                          config.CLEAN_REPAIR_LOGS_EVAL_FILE,
                          config.CLEAN_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, keep_default_na=False, na_values=[""],
                                            low_memory=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The cleaned extracts could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on cleaning, features and models reads the files the data "
            "pipeline writes, so the pipeline has to have run and written all five."
        )
    return loaded


@pytest.fixture(scope="session")
def raw_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.RAW_REPORT_HISTORY_TRAIN_FILE,
                          config.RAW_REPORT_HISTORY_EVAL_FILE,
                          config.RAW_REPAIR_LOGS_TRAIN_FILE,
                          config.RAW_REPAIR_LOGS_EVAL_FILE,
                          config.RAW_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The raw extracts supplied with the project could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The data folder has to stay as it was supplied."
        )
    return loaded


@pytest.fixture(scope="session")
def saved_models():
    from config import config

    loaded = {"problem": None}
    try:
        from modeling.model_registry import load_feature_schema, load_model

        loaded.update({
            "repeat": load_model(config.REPEAT_MODEL_FILE),
            "crew_hours": load_model(config.CREW_HOURS_MODEL_FILE),
            "sentence": load_model(config.SENTENCE_MODEL_FILE),
            "repeat_schema": list(load_feature_schema(config.REPEAT_SCHEMA_FILE)),
            "crew_hours_schema": list(load_feature_schema(config.CREW_HOURS_SCHEMA_FILE)),
        })
        for name, label, method_name in [("repeat", "repeat", "predict_proba"),
                                         ("crew_hours", "crew hours", "predict"),
                                         ("sentence", "sentence", "predict")]:
            assert callable(getattr(loaded[name], method_name, None)), (
                "load_model returned "
                + type(loaded[name]).__name__
                + " for the "
                + label
                + " model, where it returns the fitted model training saved, which has "
                + method_name
                + "."
            )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The saved models and their feature schemas could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Model training saves them, and every check on the models scores what "
            "training saved."
        )
    return loaded


@pytest.fixture(scope="session")
def live_files():
    from config import config

    loaded = {"problem": None}
    try:
        from data_pipeline.dataset_loader import load_raw_dataset
        from tools.report_data_tools import (fetch_report, load_open_reports, load_report_documents,
                                             load_street_works_register)

        loaded.update({
            "raw_reports": load_raw_dataset(config.OPEN_DEFECT_REPORTS_FILE),
            "reports": load_open_reports(),
            "documents": load_report_documents(),
            "register": load_street_works_register(),
        })
        for name, column_name in [("raw_reports", config.REPORT_REFERENCE_COLUMN),
                                  ("reports", config.DEFECT_TYPE_COLUMN),
                                  ("documents", config.DOCUMENT_TEXT_COLUMN),
                                  ("register", config.PERMIT_STATUS_COLUMN)]:
            assert column_name in getattr(loaded[name], "columns", []), (
                "The live "
                + name.replace("_", " ")
                + " came back as "
                + type(loaded[name]).__name__
                + " where they are a table holding the column "
                + column_name
                + "."
            )
        first_reference = loaded["reports"][config.REPORT_REFERENCE_COLUMN].iloc[0]
        first_report = fetch_report(loaded["reports"], first_reference)
        assert isinstance(first_report, dict) and first_report.get(
            config.REPORT_REFERENCE_COLUMN) == first_reference, (
            "fetch_report returned "
            + str(first_report)[:200]
            + " for "
            + str(first_reference)
            + ", where it returns that report's answers keyed by their columns."
        )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The live files could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The queue, the screening of a live report and the workflow all start from "
            "the open defect reports, the documents attached to them and the street works "
            "register, read through the same cleaning as the extracts."
        )
    return loaded


@pytest.fixture(scope="session")
def knowledge_collection():
    from config import config

    opened = {"problem": None}
    try:
        from knowledge.vector_index import open_vector_index

        opened["collection"] = open_vector_index(config.VECTOR_STORE_DIRECTORY)
        assert callable(getattr(opened["collection"], "query", None)), (
            "open_vector_index returned "
            + type(opened["collection"]).__name__
            + " where it returns the collection the passages are stored in."
        )
    except (Exception, SystemExit) as error:
        opened["problem"] = (
            "The knowledge index could not be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Knowledge indexing builds it, and every check on retrieval reads it."
        )
    return opened


@pytest.fixture(scope="session")
def document_texts():
    from config import config

    read = {"problem": None}
    try:
        from knowledge.pdf_extraction import extract_text_from_directory

        read["documents"] = extract_text_from_directory(config.DOCS_DIRECTORY)
        assert isinstance(read["documents"], list) and len(read["documents"]) > 0, (
            "extract_text_from_directory returned "
            + type(read["documents"]).__name__
            + " where it returns one entry for each PDF in the folder."
        )
        unkeyed = [entry for entry in read["documents"]
                   if not isinstance(entry, dict)
                   or not {"document_name", "document_text"} <= set(entry)]
        assert len(unkeyed) == 0, (
            str(len(unkeyed))
            + " entries read from the document folder do not name their document and carry its "
            "text, for example "
            + str(unkeyed[:1])[:200]
            + "."
        )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The document folder could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the text of the sixteen "
            "documents."
        )
    return read


@pytest.fixture(scope="session")
def document_passages(document_texts):
    cut = {"problem": document_texts["problem"]}
    if cut["problem"] is not None:
        return cut
    try:
        from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

        cut["documents"] = document_texts["documents"]
        cut["sections"] = split_documents_into_sections(cut["documents"])
        cut["chunks"] = build_chunk_records(cut["sections"])
        assert isinstance(cut["sections"], list) and isinstance(cut["chunks"], list), (
            "Splitting the documents returned "
            + type(cut["sections"]).__name__
            + " and cutting the sections "
            + type(cut["chunks"]).__name__
            + ", where each returns a list of entries."
        )
        for entry_name, entries, keys in [
                ("section", cut["sections"], ["document_name", "section_number", "section_text"]),
                ("passage", cut["chunks"], ["chunk_id", "chunk_text", "document_name"])]:
            unkeyed = [entry for entry in entries
                       if not isinstance(entry, dict) or not set(keys) <= set(entry)]
            assert len(entries) > 0 and len(unkeyed) == 0, (
                "Every "
                + entry_name
                + " carries "
                + str(keys)
                + ", but "
                + str(len(unkeyed))
                + " of the "
                + str(len(entries))
                + " do not, for example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        cut["problem"] = (
            "The documents could not be split into sections and cut into passages ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the passages the sixteen "
            "documents are cut into."
        )
    return cut


@pytest.fixture(scope="session")
def defect_standards(document_texts):
    from config import config

    read = {"problem": document_texts["problem"], "standards": {}}
    if read["problem"] is not None:
        return read
    try:
        from knowledge.standard_reading import read_defect_standard

        for defect_type in config.DEFECT_STANDARD_DOCUMENTS:
            read["standards"][defect_type] = read_defect_standard(
                defect_type, document_texts["documents"])
            assert isinstance(read["standards"][defect_type], dict) and isinstance(
                read["standards"][defect_type].get("requirements"), list), (
                "read_defect_standard returned "
                + str(read["standards"][defect_type])[:200]
                + " for "
                + defect_type
                + ", where it returns the standard with its requirements."
            )
            unkeyed = [requirement for requirement
                       in read["standards"][defect_type]["requirements"]
                       if not isinstance(requirement, dict)
                       or not {"requirement_kind", "requirement_name", "requirement_sentence",
                               "rules"} <= set(requirement)]
            assert len(unkeyed) == 0, (
                "The standard read for "
                + defect_type
                + " holds requirements without their kind, name, sentence and rules, for "
                "example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The defect standards could not be read from their documents ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Each report is screened against the requirements of its own defect type's "
            "standard, and of its linked type's, read from the sections of that standard's own "
            "document."
        )
    return read


@pytest.fixture(scope="session")
def invented_standards():
    from config import config

    invented = {
        config.CARRIAGEWAY: [
            (config.PHOTOGRAPH_REQUIREMENT, "Picture",
             "A photograph of the defect is attached to the report."),
            (config.LOCATION_FIXED_REQUIREMENT, "Fixed spot",
             "The location of the defect is fixed to a house number, a lamp column number or a "
             "road junction."),
            (config.DEFECT_DEPTH_REQUIREMENT, "Step depth",
             "The defect has a depth or a vertical step of at least 35 millimetres."),
            (config.DEFECT_WIDTH_REQUIREMENT, "Size",
             "The defect measures at least 120 millimetres and at most 1300 millimetres "
             "across.")],
        config.FOOTWAY: [
            (config.LOCATION_FIXED_REQUIREMENT, "Fixed spot",
             "The location of the defect is fixed to a house number, a lamp column number or a "
             "road junction."),
            (config.DEFECT_DEPTH_REQUIREMENT, "Trip height",
             "The defect has a depth or a vertical step of at least 15 millimetres."),
            (config.DEFECT_WIDTH_REQUIREMENT, "Size",
             "The defect measures at least 110 millimetres and at most 1100 millimetres "
             "across."),
            (config.PHOTOGRAPH_REQUIREMENT, "Picture",
             "A photograph of the defect is attached to the report.")],
        config.DRAINAGE: [
            (config.LOCATION_FIXED_REQUIREMENT, "Fixed spot",
             "The location of the defect is fixed to a house number, a lamp column number or a "
             "road junction."),
            (config.PHOTOGRAPH_REQUIREMENT, "Picture",
             "A photograph of the defect is attached to the report."),
            (config.WATER_DEPTH_REQUIREMENT, "Water level",
             "The standing water is at least 45 millimetres deep."),
            (config.WATER_REACH_REQUIREMENT, "Spread",
             "Where the water stands on the carriageway, it reaches at least 1.3 metres out from "
             "the kerb.")],
        config.VEGETATION: [
            (config.HIGHWAY_LAND_REQUIREMENT, "Council land",
             "The tree or hedge grows on highway land."),
            (config.LOCATION_FIXED_REQUIREMENT, "Fixed spot",
             "The location of the defect is fixed to a house number, a lamp column number or a "
             "road junction."),
            (config.CLEAR_HEIGHT_REQUIREMENT, "Headroom",
             "The clear height under the overhang is less than 2.2 metres over a footway, or "
             "less than 4.9 metres over a carriageway.")],
        config.STREET_LIGHTING: [
            (config.LOCATION_FIXED_REQUIREMENT, "Fixed spot",
             "The location of the defect is fixed to a house number, a lamp column number or a "
             "road junction."),
            (config.COLUMN_LEAN_REQUIREMENT, "Lean",
             "Where the column leans, it leans at least 4.5 degrees from upright."),
            (config.COLUMN_NUMBER_REQUIREMENT, "Column plate",
             "The lamp column number is given on the report.")],
    }
    documents = []
    for position, (defect_type, requirements) in enumerate(invented.items(), start=1):
        requirement_lines = []
        for number, requirement in enumerate(requirements, start=1):
            requirement_lines.append(str(number) + "\n" + requirement[1] + ". " + requirement[2])
        documents.append({
            "document_name": config.DEFECT_STANDARD_DOCUMENTS[defect_type],
            "document_text": (
                "WRENFORD BOROUGH COUNCIL\nDefect Standard: "
                + defect_type
                + "\nDocument Standard 9"
                + str(position)
                + " | Issue 1 | Owner: Highways Asset Management | Review cycle: every twelve "
                "months\n1. Purpose and scope\nThis standard is written for the checks alone. Its "
                "figures belong to no standard of Wrenford Borough Council.\n2. What this standard "
                "covers\nThis standard covers the defects its defect type names, on any street the "
                "council maintains.\nThe defect type is "
                + defect_type
                + ".\n3. Who reports\nA report is made by whoever sees the defect.\n"
                "4. Requirements\n"
                "Every requirement below must be Met for a Highways Inspector to order a repair "
                "straight from the report. Each requirement is stated once, in one sentence, and a "
                "report is screened against all of them, in this order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n5. Reading the requirements\nEach requirement is Met, Not met or Not "
                "documented on the report as the Report Rules read it.\n"),
        })
    read = {"problem": None, "invented": invented, "documents": documents, "standards": {}}
    try:
        from knowledge.standard_reading import read_defect_standard

        for defect_type in invented:
            read["standards"][defect_type] = read_defect_standard(defect_type, documents)
            standard = read["standards"][defect_type]
            assert (isinstance(standard, dict) and isinstance(standard.get("requirements"), list)
                    and all(isinstance(requirement, dict)
                            for requirement in standard["requirements"])), (
                "read_defect_standard returned "
                + str(standard)[:200]
                + " for "
                + defect_type
                + ", where it returns the standard with its requirements, each a record of its "
                "own."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "Five standards written for the checks, every figure different from Wrenford's, could "
            "not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). They are laid out as Wrenford's standards are, and each defect type is read "
            "from its own standard among them with the project's own reader."
        )
    return read


@pytest.fixture(scope="session")
def service_client():
    service = {"problem": None}
    try:
        from fastapi.testclient import TestClient

        from api.routes import service_application

        service["client"] = TestClient(service_application, raise_server_exceptions=False)
    except (Exception, SystemExit) as error:
        service["problem"] = (
            "The REST service could not be started ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on sign-in, access, the queue, the workflow and the chat goes "
            "through it."
        )
    return service


@pytest.fixture(scope="session")
def staff(service_client):
    from config import config

    signed_in = {"problem": service_client["problem"], "inspectors": {}}
    if signed_in["problem"] is not None:
        return signed_in
    try:
        coordinators = [account for account in config.SEED_USER_ACCOUNTS
                        if account["role"] == config.ROLE_STREET_CARE_COORDINATOR]
        advisers = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_CUSTOMER_SERVICES_ADVISER]
        inspectors = [account for account in config.SEED_USER_ACCOUNTS
                      if account["role"] == config.ROLE_HIGHWAYS_INSPECTOR]
        for staff_key, account in ([("coordinator", coordinators[0]), ("adviser", advisers[0])]
                                   + [(account["username"], account) for account in inspectors]):
            reply = service_client["client"].post(
                "/auth/signin",
                json={"username": account["username"], "password": account["password"]})
            assert reply.status_code == 200, (
                "Signing in as "
                + account["username"]
                + " with the password the configuration seeds returned "
                + str(reply.status_code)
                + ", so no check that needs a signed-in "
                + account["role"]
                + " can run."
            )
            signed = dict(reply.json(), headers={
                config.SESSION_TOKEN_HEADER: reply.json()["session_token"]})
            if account["role"] == config.ROLE_HIGHWAYS_INSPECTOR:
                signed_in["inspectors"].setdefault(account["district"], []).append(signed)
            else:
                signed_in[staff_key] = signed
    except (Exception, SystemExit) as error:
        signed_in["problem"] = (
            "The seeded staff could not be signed in ("
            + type(error).__name__
            + ": "
            + str(error)[:400]
            + "). The accounts are the ones the configuration seeds through the database "
            "setup, and a sign-in answers with the user's display name, role, district and a "
            "session token."
        )
    return signed_in


@pytest.fixture(scope="session", autouse=True)
def language_model_requests(request):
    from concurrent.futures import ThreadPoolExecutor

    from config import config

    question = "What does the footway defect standard ask for before a repair is ordered?"
    started = {"review": {"problem": "The review run was never started."},
               "chat": {"problem": "The chat question was never asked.", "question": question,
                        "latest_turn_before": 0}}
    wanted = set()
    for item in request.session.items:
        wanted.update({"saved_run", "chat_replies"} & set(item.fixturenames))
    with ThreadPoolExecutor(max_workers=2) as executor:
        service_client = {"problem": "The REST service was never started."}
        staff = {"problem": None}
        if wanted:
            service_client = request.getfixturevalue("service_client")
            staff = request.getfixturevalue("staff")
        service_ready = service_client["problem"] is None and staff["problem"] is None
        if "saved_run" in wanted and service_ready:
            client = service_client["client"]
            coordinator = staff["coordinator"]
            review = {"problem": None,
                      "coordinator_request": ("Are the requirements of the standard met on the "
                                              "report, and does any document disagree with "
                                              "it?")}
            try:
                queue = client.get("/queue",
                                   params={"defect_type": config.QUEUE_DEFECT_TYPE_ALL},
                                   headers=coordinator["headers"])
                assert queue.status_code == 200 and isinstance(queue.json(), list), (
                    "The report queue returned "
                    + str(queue.status_code)
                    + " with "
                    + queue.text[:300]
                    + ", so no report could be chosen to review."
                )
                reviewed = client.get("/reviews",
                                      params={"status_value": config.REVIEW_STATUS_OPTIONS},
                                      headers=coordinator["headers"])
                assert reviewed.status_code == 200, (
                    "Listing the report reviews returned "
                    + str(reviewed.status_code)
                    + " with "
                    + reviewed.text[:300]
                    + ", so a report nobody has reviewed yet could not be chosen."
                )
                already_reviewed = {(record["report_reference"], record["revision_number"])
                                    for record in reviewed.json()}
                choices = []
                for queue_row in queue.json():
                    if (queue_row["assessment"] == config.DECISION_ORDER
                            and (queue_row["report_reference"], queue_row["revision_number"])
                            not in already_reviewed):
                        choices.append((queue_row["document_count"],
                                        queue_row["report_reference"], queue_row["street_name"],
                                        queue_row["revision_number"]))
                assert len(choices) > 0, (
                    "No report in the queue whose assessment is ORDER REPAIR is left "
                    "unreviewed, so the one review run cannot be opened. Start again from an "
                    "empty database."
                )
                chosen = sorted(choices)[0]
                review["report_reference"] = chosen[1]
                review["revision_number"] = chosen[3]
                appointed = [district for district, streets
                             in config.INSPECTION_DISTRICT_STREETS.items() if chosen[2] in streets]
                assert len(appointed) == 1 and len(staff["inspectors"].get(appointed[0],
                                                                           [])) >= 2, (
                    "The street of "
                    + chosen[1]
                    + ", "
                    + str(chosen[2])
                    + ", belongs to the districts "
                    + str(appointed)
                    + ", where every street belongs to one district with two Highways Inspectors "
                    "appointed for it."
                )
                review["opener"] = staff["inspectors"][appointed[0]][0]
                review["decider"] = staff["inspectors"][appointed[0]][1]
                review["outsider"] = [signed for district, signed_list
                                      in staff["inspectors"].items()
                                      if district != appointed[0] for signed in signed_list][0]
                reply = client.get("/queue/" + review["report_reference"],
                                   headers=coordinator["headers"])
                assert reply.status_code == 200, (
                    "The detail of "
                    + review["report_reference"]
                    + " returned "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:300]
                    + ", so what the review starts from cannot be read."
                )
                review["detail"] = reply.json()
                from persistence.database import session_factory
                from persistence.models import AgentRun

                with session_factory() as session:
                    review["runs_before"] = session.query(AgentRun).filter(
                        AgentRun.report_reference == review["report_reference"]).count()
                review["opened"] = executor.submit(
                    client.post, "/reviews",
                    json={"report_reference": review["report_reference"],
                          "coordinator_request": review["coordinator_request"]},
                    headers=review["opener"]["headers"])
            except (Exception, SystemExit) as error:
                review["problem"] = (
                    "The one review run that the agent, workflow and record checks read could "
                    "not be opened ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:600]
                    + ")."
                )
            started["review"] = review
        if "chat_replies" in wanted and service_ready:
            client = service_client["client"]
            asker = staff["coordinator"]
            chat = {"problem": None, "question": question, "latest_turn_before": 0}
            try:
                earlier = client.get("/chat/turns", params={"turn_limit": 1},
                                     headers=asker["headers"])
                if earlier.status_code == 200:
                    chat["latest_turn_before"] = max([turn["turn_id"] for turn in earlier.json()]
                                                     + [0])
                chat["asked"] = executor.submit(client.post, "/chat/ask",
                                                json={"question": question},
                                                headers=asker["headers"])
            except (Exception, SystemExit) as error:
                chat["problem"] = (
                    "The chat question could not be asked ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:300]
                    + ")."
                )
            started["chat"] = chat
        yield started


@pytest.fixture(scope="session")
def saved_run(language_model_requests, service_client, staff):
    run = {"problem": service_client["problem"] or staff["problem"], "decision_problem": None}
    if run["problem"] is not None:
        return run
    from config import config

    client = service_client["client"]
    coordinator = staff["coordinator"]
    review = language_model_requests["review"]
    run["problem"] = review["problem"]
    if run["problem"] is not None:
        return run
    for key in ["coordinator_request", "report_reference", "revision_number", "detail",
                "runs_before", "opener", "decider", "outsider"]:
        run[key] = review[key]
    opener, decider, outsider = run["opener"], run["decider"], run["outsider"]
    run["decision_reason"] = ("Every requirement of the standard is met on the report and no "
                              "coordination trigger applies, so the repair may be ordered.")
    for name in ["calls_at_gate", "short_reason", "unknown_decision", "decided", "final_calls",
                 "final_record", "second_review", "short_request"]:
        run[name] = {"status": None, "body": "never requested"}
    for name in ["own_decision", "outside_decision"]:
        run[name] = {"status": None, "body": "never requested", "calls_before": None,
                     "calls_after": None}
    run["run_record"] = {"problem": "never read"}
    run["heard_at_gate"] = {"problem": "never asked"}
    run["heard_after"] = {"problem": "never asked"}
    try:
        opened = review["opened"].result()
        assert opened.status_code == 201, (
            "Opening a review of "
            + run["report_reference"]
            + " returned "
            + str(opened.status_code)
            + " with "
            + opened.text[:400]
            + ", so the workflow never reached a Highways Inspector."
        )
        run["at_gate"] = opened.json()
        review_id = run["at_gate"]["review_id"]
    except (Exception, SystemExit) as error:
        run["problem"] = (
            "The one review run that the agent, workflow and record checks read could not be "
            "opened ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    calls_path = "/reviews/" + review_id + "/calls"
    decision_path = "/reviews/" + review_id + "/decision"
    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["calls_at_gate"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["calls_at_gate"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from chat.record_answer import answer_report_record
        from persistence.database import session_factory
        from tools.report_data_tools import load_open_reports

        open_reports = load_open_reports()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("adviser_status", staff["adviser"], config.CHAT_TYPE_REPORT_STATUS),
                    ("adviser_prediction", staff["adviser"], config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", coordinator, config.CHAT_TYPE_REPORT_STATUS)]:
                heard[heard_key] = answer_report_record(
                    session, asker, open_reports, question_type, run["report_reference"])
        run["heard_at_gate"] = heard
    except (Exception, SystemExit) as error:
        run["heard_at_gate"] = {"problem": (
            "What is heard about the report while the review waits could not be worked out ("
            + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}

    try:
        for refused_key, refused_by in [("own_decision", opener), ("outside_decision", outsider)]:
            before = client.get(calls_path, headers=coordinator["headers"]).json()
            reply = client.post(decision_path, json={"decision": config.DECISION_ORDER,
                                                     "decision_reason": run["decision_reason"]},
                                headers=refused_by["headers"])
            after = client.get(calls_path, headers=coordinator["headers"]).json()
            run[refused_key] = {"status": reply.status_code, "body": reply.json(),
                                "calls_before": before.get("call_count"),
                                "calls_after": after.get("call_count")}
        for refusal_key, payload in [
                ("short_reason", {"decision": config.DECISION_ORDER, "decision_reason": "Fine."}),
                ("unknown_decision", {"decision": "PATCH IT",
                                      "decision_reason": run["decision_reason"]})]:
            reply = client.post(decision_path, json=payload, headers=decider["headers"])
            run[refusal_key] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post(decision_path, json={"decision": config.DECISION_ORDER,
                                                 "decision_reason": run["decision_reason"]},
                            headers=decider["headers"])
        run["decided"] = {"status": reply.status_code, "body": reply.json()}
        assert reply.status_code == 200, (
            "Ordering the repair as "
            + decider["username"]
            + ", a Highways Inspector appointed for the street who did not open the review, "
            "returned "
            + str(reply.status_code)
            + " with "
            + reply.text[:400]
            + "."
        )
    except (Exception, SystemExit) as error:
        run["decision_problem"] = (
            "The decision on the one review run could not be recorded ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )

    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["final_calls"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/reviews", params={"status_value": config.REVIEW_STATUS_OPTIONS},
                           headers=coordinator["headers"])
        matching = [record for record in reply.json() if record["review_id"] == review_id]
        run["final_record"] = {"status": reply.status_code, "body": matching}
        reply = client.post("/reviews",
                            json={"report_reference": run["report_reference"],
                                  "coordinator_request": run["coordinator_request"]},
                            headers=coordinator["headers"])
        run["second_review"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post("/reviews",
                            json={"report_reference": run["report_reference"],
                                  "coordinator_request": "ok"},
                            headers=coordinator["headers"])
        run["short_request"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["final_record"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            stored_run = session.get(AgentRun, run["at_gate"]["run_id"])
            run["run_record"] = {"problem": None, "status": stored_run.status,
                                 "llm_call_count": stored_run.llm_call_count,
                                 "thread_id": stored_run.thread_id,
                                 "runs_opened": session.query(AgentRun).filter(
                                     AgentRun.report_reference
                                     == run["report_reference"]).count()
                                 - run["runs_before"]}
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": (
            "The run's own record could not be read (" + type(error).__name__ + ": "
            + str(error)[:300] + ")."
        )}

    try:
        from chat.record_answer import answer_report_record
        from persistence.database import session_factory
        from tools.report_data_tools import load_open_reports

        open_reports = load_open_reports()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("adviser_status", staff["adviser"], config.CHAT_TYPE_REPORT_STATUS),
                    ("adviser_prediction", staff["adviser"], config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", opener, config.CHAT_TYPE_OUTCOME_PREDICTION)]:
                heard[heard_key] = answer_report_record(
                    session, asker, open_reports, question_type, run["report_reference"])
        run["heard_after"] = heard
    except (Exception, SystemExit) as error:
        run["heard_after"] = {"problem": (
            "What is heard about the report after the decision could not be worked out ("
            + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}
    return run


@pytest.fixture(scope="session")
def chat_replies(language_model_requests, service_client, staff):
    chat = language_model_requests["chat"]
    replies = {"problem": service_client["problem"] or staff["problem"] or chat["problem"],
               "standard_question": chat["question"],
               "standard": {"status": None, "body": "never asked"},
               "turns": {"status": None, "body": "never read"},
               "latest_turn_before": chat["latest_turn_before"]}
    if replies["problem"] is not None:
        return replies
    client = service_client["client"]
    asker = staff["coordinator"]
    replies["asker"] = asker["username"]
    replies["someone_else"] = staff["adviser"]["username"]
    try:
        reply = chat["asked"].result()
        replies["standard"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/chat/turns", params={"turn_limit": 5}, headers=asker["headers"])
        replies["turns"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        replies["standard"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    return replies


def test_cleaning_keeps_latest_revision_once(raw_frames, processed_frames, tmp_path):
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    try:
        cleaned_again = clean_history_dataset(
            load_raw_dataset(config.RAW_REPORT_HISTORY_EVAL_FILE))
    except Exception as error:
        cleaned_again = error
    assert isinstance(cleaned_again, pd.DataFrame), (
        "clean_history_dataset gave "
        + repr(cleaned_again)[:200]
        + " for "
        + config.RAW_REPORT_HISTORY_EVAL_FILE
        + ", where it returns the cleaned table."
    )
    rebuilt_path = str(tmp_path / "report_history_eval_cleaned_again.csv")
    save_processed_dataset(cleaned_again, rebuilt_path)
    assert os.path.isfile(rebuilt_path), (
        "save_processed_dataset wrote nothing to "
        + rebuilt_path
        + ", where it writes the table it is given to the path it is given."
    )
    rebuilt = pd.read_csv(rebuilt_path, keep_default_na=False, na_values=[""], low_memory=False)
    saved = processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE]
    assert rebuilt.shape == saved.shape and rebuilt.equals(saved), (
        "Cleaning "
        + config.RAW_REPORT_HISTORY_EVAL_FILE
        + " again gives "
        + str(rebuilt.shape)
        + " rows and columns where "
        + config.CLEAN_REPORT_HISTORY_EVAL_FILE
        + " holds "
        + str(saved.shape)
        + ", or the same shape with different values. Every check reads the cleaned files, "
        "so they have to be what the cleaning code produces now: run the data pipeline "
        "again after changing it."
    )
    for clean_path in [config.CLEAN_REPORT_HISTORY_TRAIN_FILE,
                       config.CLEAN_REPORT_HISTORY_EVAL_FILE]:
        clean_frame = processed_frames[clean_path]
        repeated = int(clean_frame[config.REPORT_REFERENCE_COLUMN].duplicated().sum())
        assert repeated == 0, (
            clean_path
            + " still holds "
            + str(repeated)
            + " rows whose report reference appears on another row. A report sent again is "
            "one report, and a model fitted on it twice weighs it twice."
        )
        outcomes = sorted(str(value) for value in
                          clean_frame[config.HISTORY_TARGET_COLUMN].dropna().unique())
        unanswered = int(clean_frame[config.HISTORY_TARGET_COLUMN].isna().sum())
        assert unanswered == 0 and outcomes == sorted(config.REPEAT_TARGET_ENCODING), (
            "The reported again column of "
            + clean_path
            + " holds "
            + str(unanswered)
            + " blanks and the answers "
            + str(outcomes[:6])
            + " where every spelling is folded onto "
            + str(sorted(config.REPEAT_TARGET_ENCODING))
            + " and no row is kept without one."
        )
    raw_frame = raw_frames[config.RAW_REPORT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    templates = raw_frame[answered].drop_duplicates(config.REPORT_REFERENCE_COLUMN).head(2)
    rows = pd.concat([templates.iloc[[0]]] * 3 + [templates.iloc[[1]]] * 2
                     + [templates.iloc[[0]]] * 2, ignore_index=True)
    rows[config.REPORT_REFERENCE_COLUMN] = ["SC-950001", "SC0950001", "SC0950001", "950002",
                                            "950002", "SC0950003", "SC-950003"]
    rows[config.REVISION_NUMBER_COLUMN] = ["1", "2", "2", "1", "1", "1", "2"]
    rows[config.PLANNED_CREW_HOURS_COLUMN] = ["1.1", "1.2", "1.2", "1.3", "1.3", "1.4", "1.5"]
    rows.loc[6, config.HISTORY_TARGET_COLUMN] = "N/A"
    planted_path = str(tmp_path / "planted_revisions.csv")
    rows.iloc[[2, 6, 3, 0, 5, 4, 1]].to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    needed = [config.REPORT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
              config.PLANNED_CREW_HOURS_COLUMN]
    assert isinstance(cleaned, pd.DataFrame) and set(needed) <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for seven planted rows, where it returns the cleaned table with the columns "
        + str(needed)
        + "."
    )
    got = sorted(zip(cleaned[config.REPORT_REFERENCE_COLUMN].map(str),
                     cleaned[config.REVISION_NUMBER_COLUMN].map(str),
                     cleaned[config.PLANNED_CREW_HOURS_COLUMN].map(str), strict=True))
    expected = [("SC0950001", "2", "1.2"), ("SC0950002", "1", "1.3")]
    assert got == expected, (
        "Seven planted rows - a report written SC-950001 at revision 1 and SC0950001 at "
        "revision 2, its second revision repeated exactly and standing first in the file; a "
        "report written 950002 repeated exactly; and a report answered at revision 1, written "
        "SC0950003, whose revision 2, written SC-950003, records no reported again answer yet - "
        "each telling its revisions apart by its planned crew hours, cleaned to "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A row repeated exactly is one row, the highest revision is the report, and a "
        "report whose highest revision records no reported again answer is not kept: the "
        "answer is never taken from an earlier revision."
    )


def test_readings_made_standard_units(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_frame = raw_frames[config.RAW_REPORT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.REPORT_REFERENCE_COLUMN).head(1)
    depth, width, water = "defect_depth_mm", "defect_width_mm", "water_depth_mm"
    reach, clear = config.WATER_REACH_COLUMN, config.CLEAR_HEIGHT_COLUMN
    lean, height = config.COLUMN_LEAN_COLUMN, config.COLUMN_HEIGHT_COLUMN
    planned = config.PLANNED_CREW_HOURS_COLUMN
    planted = [(depth, "45", 45), (depth, "4.5 cm", 45), (depth, "3.95 cm", 40),
               (depth, "3.94 cm", 39), (depth, "1.75 in", 44), (depth, "0.8 in", 20),
               (depth, "1.25 in", 32), (depth, "2.25 in", 57), (width, "250", 250),
               (width, "25.8 cm", 258), (width, "9.84 in", 250), (width, "59.05 in", 1500),
               (water, "60", 60), (water, "6 cm", 60), (water, "2.36 in", 60),
               (water, "1.95 in", 50), (reach, "1.2", 1.2), (reach, "1.2 m", 1.2),
               (reach, "4 ft", 1.2), (reach, "3.4 ft", 1.0), (reach, "16.4 ft", 5.0),
               (reach, "1.25", 1.3), (clear, "2.4", 2.4), (clear, "2.4 m", 2.4),
               (clear, "7.9 ft", 2.4), (clear, "2.35 m", 2.4), (clear, "2.34", 2.3),
               (height, "8.5", 8.5), (height, "8.5 m", 8.5), (height, "26 ft", 7.9),
               (height, "28 ft", 8.5), (lean, "6.5", 6.5), (lean, "6.5 deg", 6.5),
               (lean, "4.95", 5.0), (lean, "4.94 deg", 4.9), (planned, "2.5", 2.5),
               (planned, "150 min", 2.5), (planned, "2.5 hrs", 2.5), (planned, "1 hr", 1.0),
               (planned, "45 min", 0.8), (planned, "261 min", 4.4), (planned, "4.35 hrs", 4.4)]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, planted_reading in enumerate(planted):
        rows.loc[position, config.REPORT_REFERENCE_COLUMN] = "SC09" + str(60001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        rows.loc[position, planted_reading[0]] = planted_reading[1]
    planted_path = str(tmp_path / "planted_readings.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.REPORT_REFERENCE_COLUMN}.union(
        planted_reading[0] for planted_reading in planted)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for "
        + str(len(planted))
        + " planted readings, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.REPORT_REFERENCE_COLUMN)
    wrong = []
    for position, (column_name, written, wanted) in enumerate(planted):
        reference = "SC09" + str(60001 + position)
        if reference not in cleaned.index:
            wrong.append(column_name + " " + repr(written) + " lost its row")
            continue
        got = cleaned.loc[reference, column_name]
        if pd.isna(got) or got != wanted:
            wrong.append(column_name + " " + repr(written) + " read as " + str(got) + ", not "
                         + str(wanted))
    assert len(wrong) == 0, (
        "Readings are not read as the Report Rules set out: "
        + "; ".join(wrong[:6])
        + ". Each reading is made its unit - a depth, a width and standing water whole "
        "millimetres, centimetres times 10 and inches times 25.4; a reach, a clear height and a "
        "column height metres to one decimal, feet times 0.3048; a lean degrees to one decimal; "
        "and crew hours hours to one decimal, minutes divided by 60 - converted exactly and "
        "rounded half up once, so 3.95 cm is 40 mm and 45 min is 0.8 hours."
    )
    for column_name, written in [(depth, "45 mm"), (depth, "4.5cm"), (depth, "1.75 inches"),
                                 (depth, "1.75in"), (width, "25.8 cms"), (water, "6 cm deep"),
                                 (reach, "1.2m"), (reach, "4 feet"), (reach, "4ft"),
                                 (clear, "2.4 metres"), (height, "26 foot"),
                                 (lean, "6.5deg"), (lean, "6.5 degrees"), (planned, "150min"),
                                 (planned, "2.5 hours"), (planned, "2.5 h")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_reading.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A reading in no notation the three channels write - another unit or another "
            "spacing - is refused with a ValueError naming the column and the value as written, "
            "never guessed at or left blank."
        )


def test_references_types_and_places_canonical(raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_frame = raw_frames[config.RAW_REPORT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.REPORT_REFERENCE_COLUMN).head(1)
    cw, fw, dr = config.CARRIAGEWAY, config.FOOTWAY, config.DRAINAGE
    vg, sl, unlinked = config.VEGETATION, config.STREET_LIGHTING, config.NO_LINKED_DEFECT
    ts, lr = config.TRAFFIC_SENSITIVE, config.LOCAL_ROAD
    road, path = config.CARRIAGEWAY_PART, config.FOOTWAY_PART
    out, day, leaning = config.LAMP_OUT, config.DAY_BURNING, config.LEANING_COLUMN
    columns = [config.DEFECT_TYPE_COLUMN, config.LINKED_DEFECT_TYPE_COLUMN,
               config.STREET_CATEGORY_COLUMN, config.WATER_LOCATION_COLUMN,
               config.OVERHANG_LOCATION_COLUMN, config.LIGHTING_FAULT_COLUMN,
               config.STREET_NAME_COLUMN, "district"]
    planted = [
        ("SC0970001", "SC0970001", ["Carriageway", "No linked defect", "Traffic sensitive",
                                    "Carriageway", "Footway", "Lamp out", "High Street",
                                    "Town Centre"],
         [cw, unlinked, ts, road, path, out, "High Street", config.TOWN_CENTRE]),
        ("SC-970002", "SC0970002", ["Pothole or road surface", "None", "Main route", "Road",
                                    "Pavement", "Light not working", "high street",
                                    "town centre"],
         [cw, unlinked, ts, road, path, out, "High Street", config.TOWN_CENTRE]),
        ("970003", "SC0970003", ["CW", "NIL", "TS", "ROAD", "PATH", "OUT", "HIGH STREET",
                                 "TOWN CENTRE"],
         [cw, unlinked, ts, road, path, out, "High Street", config.TOWN_CENTRE]),
        ("SC0970004", "SC0970004", ["Footway", "Vegetation", "Local road", "Footway",
                                    "Carriageway", "Day burning", "Northgate", "North"],
         [fw, vg, lr, path, road, day, "Northgate", config.NORTH]),
        ("SC-970005", "SC0970005", ["Pavement defect", "Overhanging tree or hedge",
                                    "Residential street", "Pavement", "Road",
                                    "Light on in daytime", "NORTHGATE", "north"],
         [fw, vg, lr, path, road, day, "Northgate", config.NORTH]),
        ("970006", "SC0970006", ["FW", "VG", "LR", "PATH", "ROAD", "DAY", "northgate", "NORTH"],
         [fw, vg, lr, path, road, day, "Northgate", config.NORTH]),
        ("SC0970007", "SC0970007", ["Drainage", "Carriageway", "traffic sensitive",
                                    "carriageway", "footway", "Leaning column",
                                    "Riverside Walk", "South"],
         [dr, cw, ts, road, path, leaning, "Riverside Walk", config.SOUTH]),
        ("SC-970008", "SC0970008", ["Flooding or blocked drain", "Pothole or road surface",
                                    "main route", "road", "pavement", "Column leaning",
                                    "riverside walk", "SOUTH"],
         [dr, cw, ts, road, path, leaning, "Riverside Walk", config.SOUTH]),
        ("970009", "SC0970009", ["DR", "cw", "ts", "Road", "path", "LEAN", "Riverside walk",
                                 "south"],
         [dr, cw, ts, road, path, leaning, "Riverside Walk", config.SOUTH]),
        ("SC0970010", "SC0970010", ["Vegetation", "Footway", "Local Road", "Footway", "Footway",
                                    "Lamp Out", "Kiln Lane", "Villages"],
         [vg, fw, lr, path, path, out, "Kiln Lane", config.VILLAGES]),
        ("SC-970011", "SC0970011", ["Overhanging Tree Or Hedge", "Pavement Defect",
                                    "Residential Street", "PAVEMENT", "Pavement",
                                    "LIGHT NOT WORKING", "kiln lane", "villages"],
         [vg, fw, lr, path, path, out, "Kiln Lane", config.VILLAGES]),
        ("970012", "SC0970012", ["vg", "fw", "lr", "path", "PATH", "out", "KILN LANE",
                                 "Villages"],
         [vg, fw, lr, path, path, out, "Kiln Lane", config.VILLAGES]),
        ("SC0970013", "SC0970013", ["Street lighting", "Drainage", "Traffic Sensitive",
                                    "Carriageway", "Carriageway", "Leaning Column",
                                    "Clay Pit Lane", "Villages"],
         [sl, dr, ts, road, road, leaning, "Clay Pit Lane", config.VILLAGES]),
        ("SC-970014", "SC0970014", ["Street light fault", "Flooding or blocked drain",
                                    "MAIN ROUTE", "ROAD", "Road", "column leaning",
                                    "clay pit lane", "VILLAGES"],
         [sl, dr, ts, road, road, leaning, "Clay Pit Lane", config.VILLAGES]),
        ("97", "SC0000097", ["SL", "DR", "TS", "road", "road", "lean", "Clay Pit lane",
                             "villages"],
         [sl, dr, ts, road, road, leaning, "Clay Pit Lane", config.VILLAGES])]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, (written_reference, _wanted_reference, written, _wanted) in enumerate(planted):
        rows.loc[position, config.REPORT_REFERENCE_COLUMN] = written_reference
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        for column_name, value in zip(columns, written, strict=True):
            rows.loc[position, column_name] = value
    planted_path = str(tmp_path / "planted_notations.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    assert isinstance(cleaned, pd.DataFrame) and set(
            [config.REPORT_REFERENCE_COLUMN] + columns) <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for fifteen planted notations, where it returns the cleaned table."
    )
    wrong = []
    for written_reference, wanted_reference, written, wanted in planted:
        matching = cleaned[cleaned[config.REPORT_REFERENCE_COLUMN] == wanted_reference]
        if len(matching) != 1:
            wrong.append(written_reference + " is kept " + str(len(matching)) + " times as "
                         + wanted_reference)
            continue
        got = [matching[column_name].iloc[0] for column_name in columns]
        if any(pd.isna(value) for value in got) or got != wanted:
            wrong.append(str(written) + " read as " + str(got))
    assert len(wrong) == 0, (
        "References, defect types, street categories, where water stands and vegetation "
        "overhangs, lighting faults, streets and districts are not read as the Report Rules set "
        "out: "
        + "; ".join(wrong[:5])
        + ". A reference written SC and seven digits, SC- and the number or the number alone "
        "is SC and the number in seven digits; and each named answer, written as the website, "
        "the StreetCare app or the contact centre writes it, in any capitals, is its one "
        "spelling: CW and Pothole or road surface are Carriageway, None and NIL are No linked "
        "defect, Main route and TS are Traffic sensitive, Road and ROAD are Carriageway, LEAN is "
        "Leaning column, and a street or a district is its own name."
    )
    for column_name, written in [(config.REPORT_REFERENCE_COLUMN, "SC52118"),
                                 (config.REPORT_REFERENCE_COLUMN, "SC 52118"),
                                 (config.REPORT_REFERENCE_COLUMN, "SC-12345678"),
                                 (config.REPORT_REFERENCE_COLUMN, "12345678"),
                                 (config.REPORT_REFERENCE_COLUMN, "SW-52118"),
                                 (config.DEFECT_TYPE_COLUMN, "Pothole"),
                                 (config.LINKED_DEFECT_TYPE_COLUMN, "Nothing"),
                                 (config.STREET_CATEGORY_COLUMN, "Busy road"),
                                 (config.WATER_LOCATION_COLUMN, "Verge"),
                                 (config.LIGHTING_FAULT_COLUMN, "Flickering"),
                                 (config.STREET_NAME_COLUMN, "Station Rd"),
                                 ("district", "East"),
                                 ("reporter_type", "Tourist"),
                                 ("report_channel", "Email")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_notation.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value in no notation the three channels write is refused with a ValueError "
            "naming the column and the value as written."
        )
    defect_types = sorted(config.DEFECT_STANDARD_DOCUMENTS)
    for clean_path in [config.CLEAN_REPORT_HISTORY_TRAIN_FILE,
                       config.CLEAN_REPORT_HISTORY_EVAL_FILE,
                       config.CLEAN_REPAIR_LOGS_TRAIN_FILE,
                       config.CLEAN_REPAIR_LOGS_EVAL_FILE,
                       config.CLEAN_STATEMENT_SENTENCES_FILE]:
        clean_frame = processed_frames[clean_path]
        references = clean_frame[config.REPORT_REFERENCE_COLUMN].dropna().astype(str)
        stray = sorted(references[~references.str.fullmatch(
            config.CANONICAL_REPORT_REFERENCE_PATTERN)].unique())
        assert len(references) == len(clean_frame) and len(stray) == 0, (
            clean_path
            + " holds "
            + str(len(clean_frame) - len(references))
            + " blank report references and "
            + str(len(stray))
            + " that are not canonical, for example "
            + str(stray[:3])
            + "."
        )
        if config.DEFECT_TYPE_COLUMN in clean_frame.columns:
            found_types = sorted(str(value) for value in
                                 clean_frame[config.DEFECT_TYPE_COLUMN].dropna().unique())
            found_links = sorted(str(value) for value in
                                 clean_frame[config.LINKED_DEFECT_TYPE_COLUMN].dropna().unique())
            untyped = int(clean_frame[[config.DEFECT_TYPE_COLUMN,
                                       config.LINKED_DEFECT_TYPE_COLUMN]].isna().sum().sum())
            assert (found_types == defect_types and untyped == 0
                    and found_links == sorted(defect_types + [unlinked])), (
                clean_path
                + " holds the defect types "
                + str(found_types)
                + ", the linked types "
                + str(found_links)
                + " and "
                + str(untyped)
                + " blanks among them, where a defect type is one of the five, a linked type one "
                "of the five or No linked defect, and neither is ever blank."
            )


def test_answers_codes_and_register_read_canonically(raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset
    from data_pipeline.register_cleaning import clean_register_records
    from data_pipeline.repair_log_cleaning import clean_repair_log_dataset
    from data_pipeline.sentence_cleaning import clean_sentence_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    for clean_path, value_mappings in [
            (config.CLEAN_REPORT_HISTORY_TRAIN_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_REPORT_HISTORY_EVAL_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_REPAIR_LOGS_TRAIN_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_REPAIR_LOGS_EVAL_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_VALUE_MAPPINGS)]:
        clean_frame = processed_frames[clean_path]
        for column_name, value_mapping in value_mappings.items():
            found = set(clean_frame[column_name].dropna().astype(str))
            stray = sorted(found - set(value_mapping.values()))
            assert len(stray) == 0, (
                column_name
                + " in "
                + clean_path
                + " holds "
                + str(stray[:5])
                + ", which are not among its canonical values "
                + str(sorted(set(value_mapping.values()))[:6])
                + ". Every spelling a channel writes is folded onto one."
            )
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    fixed, photo = "location_fixed", "photograph_attached"
    land, plate = "on_highway_land", "column_number_given"
    again, outcome, cause = config.HISTORY_TARGET_COLUMN, "review_outcome", "repeat_cause"
    as_reported, inspected = config.REPAIRED_AS_REPORTED, config.REPAIRED_AFTER_INSPECTION
    coordinated, closed = config.REPAIRED_AFTER_COORDINATION, config.CLOSED_AFTER_INSPECTION
    history_planted = [
        ({fixed: "Yes", photo: "No", outcome: "Repaired as reported", again: "Yes",
          cause: "Repair failed"},
         {fixed: answered_yes, photo: answered_no, outcome: as_reported, again: answered_yes,
          cause: config.REPAIR_FAILED}),
        ({fixed: "Y", photo: "N", outcome: "Repaired", again: "Y", cause: "failed"},
         {fixed: answered_yes, photo: answered_no, outcome: as_reported, again: answered_yes,
          cause: config.REPAIR_FAILED}),
        ({fixed: "TRUE", photo: "FALSE", outcome: "REP", again: "TRUE",
          cause: "Defect returned nearby"},
         {fixed: answered_yes, photo: answered_no, outcome: as_reported, again: answered_yes,
          cause: config.DEFECT_RETURNED_NEARBY}),
        ({fixed: "true", photo: "false", outcome: "Repaired after inspection", again: "true",
          cause: "New defect nearby"},
         {fixed: answered_yes, photo: answered_no, outcome: inspected, again: answered_yes,
          cause: config.DEFECT_RETURNED_NEARBY}),
        ({fixed: "1", photo: "0", outcome: "Repaired - inspected", again: "1", cause: "NEARBY"},
         {fixed: answered_yes, photo: answered_no, outcome: inspected, again: answered_yes,
          cause: config.DEFECT_RETURNED_NEARBY}),
        ({fixed: "y", photo: "n", outcome: "REP-INS", again: "y", cause: "Regrowth"},
         {fixed: answered_yes, photo: answered_no, outcome: inspected, again: answered_yes,
          cause: config.REGROWTH}),
        ({fixed: "yes", photo: "no", outcome: "Repaired after coordination", again: "yes",
          cause: "Grown back"},
         {fixed: answered_yes, photo: answered_no, outcome: coordinated, again: answered_yes,
          cause: config.REGROWTH}),
        ({land: "YES", plate: "NO", outcome: "Repaired - coordinated", again: "YES",
          cause: "REGROWTH"},
         {land: answered_yes, plate: answered_no, outcome: coordinated, again: answered_yes,
          cause: config.REGROWTH}),
        ({land: "True", plate: "False", outcome: "rep-crd", again: "True",
          cause: "Blocked again"},
         {land: answered_yes, plate: answered_no, outcome: coordinated, again: answered_yes,
          cause: config.BLOCKED_AGAIN}),
        ({land: "Yes", plate: "No", outcome: "Closed after inspection", again: "Yes",
          cause: "REBLOCKED"},
         {land: answered_yes, plate: answered_no, outcome: closed, again: answered_yes,
          cause: config.BLOCKED_AGAIN}),
        ({land: "Y", plate: "N", outcome: "Closed - inspected", again: "Y",
          cause: "Lamp failed again"},
         {land: answered_yes, plate: answered_no, outcome: closed, again: answered_yes,
          cause: config.LAMP_FAILED_AGAIN}),
        ({fixed: "Yes", outcome: "CLS-INS", again: "Yes", cause: "Failed again"},
         {fixed: answered_yes, outcome: closed, again: answered_yes,
          cause: config.LAMP_FAILED_AGAIN}),
        ({photo: "No", outcome: "cls-ins", again: "Yes", cause: "REFAIL"},
         {photo: answered_no, outcome: closed, again: answered_yes,
          cause: config.LAMP_FAILED_AGAIN}),
        ({fixed: "Yes", outcome: "Repaired", again: "Yes", cause: "Worsened after inspection"},
         {fixed: answered_yes, outcome: as_reported, again: answered_yes,
          cause: config.WORSENED_AFTER_INSPECTION}),
        ({fixed: "No", outcome: "Repaired", again: "Yes", cause: "Got worse"},
         {fixed: answered_no, outcome: as_reported, again: answered_yes,
          cause: config.WORSENED_AFTER_INSPECTION}),
        ({fixed: "No", outcome: "REP", again: "Yes", cause: "worse"},
         {fixed: answered_no, outcome: as_reported, again: answered_yes,
          cause: config.WORSENED_AFTER_INSPECTION}),
        ({fixed: "No", outcome: "Repaired", again: "No", cause: "N/A"},
         {fixed: answered_no, outcome: as_reported, again: answered_no, cause: None}),
        ({fixed: "N", outcome: "REP", again: "false", cause: ""},
         {fixed: answered_no, outcome: as_reported, again: answered_no, cause: None}),
        ({fixed: "0", outcome: "Repaired", again: "0", cause: "-"},
         {fixed: answered_no, outcome: as_reported, again: answered_no, cause: None})]
    raw_history = raw_frames[config.RAW_REPORT_HISTORY_EVAL_FILE]
    answered = ~raw_history[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    history_template = raw_history[answered].drop_duplicates(
        config.REPORT_REFERENCE_COLUMN).head(1)
    log_template = raw_frames[config.RAW_REPAIR_LOGS_EVAL_FILE].drop_duplicates(
        config.REPORT_REFERENCE_COLUMN).head(1)
    sentence_template = raw_frames[config.RAW_STATEMENT_SENTENCES_FILE].head(1)
    code, method, traffic = "completion_code", "repair_method", "traffic_management"
    completed, overran = config.COMPLETED, config.COMPLETED_WITH_OVERRUN
    returned = config.COMPLETED_AFTER_RETURN_VISIT
    log_planted = [
        ({code: "Completed", again: "No", method: "Patch repair", traffic: "None"},
         {code: completed, again: answered_no, method: config.PATCH_REPAIR,
          traffic: config.NO_TRAFFIC_MANAGEMENT}),
        ({code: "Done", again: "false", method: "Patch", traffic: "No traffic management"},
         {code: completed, again: answered_no, method: config.PATCH_REPAIR,
          traffic: config.NO_TRAFFIC_MANAGEMENT}),
        ({code: "COMP", again: "N", method: "PATCH", traffic: "NONE"},
         {code: completed, again: answered_no, method: config.PATCH_REPAIR,
          traffic: config.NO_TRAFFIC_MANAGEMENT}),
        ({code: "Completed with overrun", again: "No", method: "Permanent reinstatement",
          traffic: "Signs only"},
         {code: overran, again: answered_no, method: config.PERMANENT_REINSTATEMENT,
          traffic: config.SIGNS_ONLY}),
        ({code: "Done - overran", again: "0", method: "Permanent", traffic: "Signs"},
         {code: overran, again: answered_no, method: config.PERMANENT_REINSTATEMENT,
          traffic: config.SIGNS_ONLY}),
        ({code: "COMP-OVR", again: "n", method: "PERM", traffic: "SIGNS"},
         {code: overran, again: answered_no, method: config.PERMANENT_REINSTATEMENT,
          traffic: config.SIGNS_ONLY}),
        ({code: "Completed after return visit", again: "Yes", method: "Slab relay",
          traffic: "Lane closure"},
         {code: returned, again: answered_yes, method: config.SLAB_RELAY,
          traffic: config.LANE_CLOSURE}),
        ({code: "Done - second visit", again: "true", method: "Relay slabs",
          traffic: "Lane closed"},
         {code: returned, again: answered_yes, method: config.SLAB_RELAY,
          traffic: config.LANE_CLOSURE}),
        ({code: "comp-rtn", again: "Y", method: "relay", traffic: "LANE"},
         {code: returned, again: answered_yes, method: config.SLAB_RELAY,
          traffic: config.LANE_CLOSURE}),
        ({code: "done", again: "FALSE", method: "Gully cleanse", traffic: "Road closure"},
         {code: completed, again: answered_no, method: config.GULLY_CLEANSE,
          traffic: config.ROAD_CLOSURE}),
        ({method: "Cleanse gully", traffic: "Road closed"},
         {method: config.GULLY_CLEANSE, traffic: config.ROAD_CLOSURE}),
        ({method: "GULLY", traffic: "CLOSED"},
         {method: config.GULLY_CLEANSE, traffic: config.ROAD_CLOSURE}),
        ({method: "Drain repair"}, {method: config.DRAIN_REPAIR}),
        ({method: "Repair drain"}, {method: config.DRAIN_REPAIR}),
        ({method: "DRAIN"}, {method: config.DRAIN_REPAIR}),
        ({method: "Cut back"}, {method: config.CUT_BACK}),
        ({method: "CUT"}, {method: config.CUT_BACK}),
        ({method: "Lamp replacement"}, {method: config.LAMP_REPLACEMENT}),
        ({method: "Replace lamp"}, {method: config.LAMP_REPLACEMENT}),
        ({method: "LAMP"}, {method: config.LAMP_REPLACEMENT}),
        ({method: "Photocell replacement"}, {method: config.PHOTOCELL_REPLACEMENT}),
        ({method: "Replace photocell"}, {method: config.PHOTOCELL_REPLACEMENT}),
        ({method: "PECU"}, {method: config.PHOTOCELL_REPLACEMENT}),
        ({method: "Column replacement"}, {method: config.COLUMN_REPLACEMENT}),
        ({method: "Replace column"}, {method: config.COLUMN_REPLACEMENT}),
        ({method: "COLUMN"}, {method: config.COLUMN_REPLACEMENT})]
    label, kind = config.SENTENCE_TARGET_COLUMN, config.DOCUMENT_TYPE_COLUMN
    sentence_planted = [
        ({label: "Measurement", kind: "Resident description"},
         {label: config.MEASUREMENT, kind: config.RESIDENT_DESCRIPTION}),
        ({label: "MEASUREMENT", kind: "CALL NOTE"},
         {label: config.MEASUREMENT, kind: config.CALL_NOTE}),
        ({label: "Measurements", kind: "photo note"},
         {label: config.MEASUREMENT, kind: config.PHOTO_NOTE}),
        ({label: "measurement", kind: "Inspection Note"},
         {label: config.MEASUREMENT, kind: config.INSPECTION_NOTE}),
        ({label: "LOCATION"}, {label: config.LOCATION}),
        ({label: "Where"}, {label: config.LOCATION}),
        ({label: "location"}, {label: config.LOCATION}),
        ({label: "PHOTOGRAPH"}, {label: config.PHOTOGRAPH}),
        ({label: "Photo"}, {label: config.PHOTOGRAPH}),
        ({label: "HAZARD"}, {label: config.HAZARD}),
        ({label: "Danger"}, {label: config.HAZARD}),
        ({label: "INCIDENT"}, {label: config.INCIDENT}),
        ({label: "Accident"}, {label: config.INCIDENT}),
        ({label: "Traffic & use"}, {label: config.TRAFFIC_AND_USE}),
        ({label: "Traffic"}, {label: config.TRAFFIC_AND_USE}),
        ({label: "traffic-and-use"}, {label: config.TRAFFIC_AND_USE}),
        ({label: "Traffic and use"}, {label: config.TRAFFIC_AND_USE}),
        ({label: "PREVIOUS REPORTS"}, {label: config.PREVIOUS_REPORTS}),
        ({label: "Previous report"}, {label: config.PREVIOUS_REPORTS}),
        ({label: "previous-reports"}, {label: config.PREVIOUS_REPORTS}),
        ({label: "Weather & water"}, {label: config.WEATHER_AND_WATER}),
        ({label: "Weather"}, {label: config.WEATHER_AND_WATER}),
        ({label: "weather-and-water"}, {label: config.WEATHER_AND_WATER}),
        ({label: "Context"}, {label: config.BACKGROUND}),
        ({label: "General"}, {label: config.BACKGROUND}),
        ({label: "background"}, {label: config.BACKGROUND})]
    raw_register = load_raw_dataset(config.STREET_WORKS_REGISTER_FILE)
    street, works, status = (config.STREET_NAME_COLUMN, config.WORKS_LOCATION_COLUMN,
                             config.PERMIT_STATUS_COLUMN)
    promoter = "works_promoter"
    register_planted = [
        ({street: "mill lane", works: "carriageway", promoter: "WATER COMPANY", status: "LIVE"},
         {street: "Mill Lane", works: config.CARRIAGEWAY_PART, promoter: "Water company",
          status: config.PERMIT_STATUS_LIVE}),
        ({street: "HAWTHORN CLOSE", works: "FOOTWAY", promoter: "gas network",
          status: "cancelled"},
         {street: "Hawthorn Close", works: config.FOOTWAY_PART, promoter: "Gas network",
          status: config.PERMIT_STATUS_CANCELLED}),
        ({street: "Kiln lane", works: "Footway", promoter: "Telecoms", status: "Live"},
         {street: "Kiln Lane", works: config.FOOTWAY_PART, promoter: "Telecoms",
          status: config.PERMIT_STATUS_LIVE})]
    register_references = [("SWP0081207", "SWP0081207"), ("SWP0081208", "SWP0081208"),
                           ("SWP0081209", "SWP0081209")]
    wrong = []
    for name, template, planted, key_column, keys, cleaner in [
            ("report history", history_template, history_planted,
             config.REPORT_REFERENCE_COLUMN,
             [("SC09" + str(81001 + position), "SC09" + str(81001 + position))
              for position in range(len(history_planted))], clean_history_dataset),
            ("repair log", log_template, log_planted, config.REPORT_REFERENCE_COLUMN,
             [("SC09" + str(82001 + position), "SC09" + str(82001 + position))
              for position in range(len(log_planted))], clean_repair_log_dataset),
            ("statement sentence", sentence_template, sentence_planted,
             config.SENTENCE_ID_COLUMN,
             [("SEN-98" + str(30001 + position), "SEN-98" + str(30001 + position))
              for position in range(len(sentence_planted))], clean_sentence_dataset),
            ("street works register", raw_register.head(1), register_planted,
             config.PERMIT_REFERENCE_COLUMN, register_references, clean_register_records)]:
        rows = pd.concat([template] * len(planted), ignore_index=True)
        for position, (written, _wanted) in enumerate(planted):
            rows.loc[position, key_column] = keys[position][0]
            if config.REVISION_NUMBER_COLUMN in rows.columns:
                rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
            for column_name, value in written.items():
                rows.loc[position, column_name] = value
        planted_path = str(tmp_path / "planted_answers.csv")
        rows.to_csv(planted_path, index=False)
        try:
            cleaned = cleaner(load_raw_dataset(planted_path))
        except Exception as error:
            cleaned = error
        read_columns = {key_column}.union(*[set(written) for written, _wanted in planted])
        assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
            "Cleaning "
            + str(len(planted))
            + " planted rows of the "
            + name
            + " gave "
            + repr(cleaned)[:200]
            + ", where it returns the cleaned table with the columns "
            + str(sorted(read_columns))
            + "."
        )
        cleaned = cleaned.set_index(key_column)
        for position, (written, wanted) in enumerate(planted):
            key = keys[position][1]
            if key not in cleaned.index:
                wrong.append("the planted " + name + " row " + key + " was lost")
                continue
            for column_name, expected in wanted.items():
                got = cleaned.loc[key, column_name]
                if (expected is None and pd.isna(got)) or (not pd.isna(got) and got == expected):
                    continue
                wrong.append(name + " " + column_name + " " + repr(written[column_name])
                             + " read as " + repr(got) + ", not " + repr(expected))
    assert len(wrong) == 0, (
        "Answers are not read as the Report Rules set out: "
        + "; ".join(wrong[:6])
        + ". Yes is Yes, Y, TRUE or 1 and no is No, N, FALSE or 0, in any capitals; a review "
        "outcome, a repeat cause, a repair method, the traffic management, a completion code, a "
        "sentence label and a document type take their one spelling whichever way a channel "
        "writes them; a blank is a blank; and the street works register's streets, works "
        "locations, promoters and statuses are read whatever their capitals."
    )
    for name, template, cleaner, column_name, written in [
            ("report history", history_template, clean_history_dataset, outcome, "Withdrawn"),
            ("report history", history_template, clean_history_dataset, cause, "Vandalism"),
            ("report history", history_template, clean_history_dataset, fixed, "Maybe"),
            ("report history", history_template, clean_history_dataset, again, "Perhaps"),
            ("repair log", log_template, clean_repair_log_dataset, code, "Abandoned"),
            ("repair log", log_template, clean_repair_log_dataset, method, "Resurfacing"),
            ("repair log", log_template, clean_repair_log_dataset, traffic, "Convoy"),
            ("repair log", log_template, clean_repair_log_dataset, again, "Unknown"),
            ("statement sentence", sentence_template, clean_sentence_dataset, label, "Rumour"),
            ("statement sentence", sentence_template, clean_sentence_dataset, kind, "Drawing"),
            ("street works register", raw_register.head(1), clean_register_records, status,
             "Closed"),
            ("street works register", raw_register.head(1), clean_register_records, street,
             "Station Rd"),
            ("street works register", raw_register.head(1), clean_register_records, promoter,
             "Cable company"),
            ("street works register", raw_register.head(1), clean_register_records,
             config.PERMIT_REFERENCE_COLUMN, "SWP-0081210")]:
        unreadable = template.copy()
        if config.REVISION_NUMBER_COLUMN in unreadable.columns:
            unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_answer.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            cleaner(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + name
            + " row whose "
            + column_name
            + " is written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value with no reading in the Report Rules is refused with a ValueError naming "
            "the column and the value as written."
        )


def test_impossible_readings_blanked_bad_keys_refused(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset
    from data_pipeline.register_cleaning import clean_register_records
    from data_pipeline.repair_log_cleaning import clean_repair_log_dataset
    from data_pipeline.sentence_cleaning import clean_sentence_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_history = raw_frames[config.RAW_REPORT_HISTORY_EVAL_FILE]
    answered = ~raw_history[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_history[answered].drop_duplicates(config.REPORT_REFERENCE_COLUMN).head(1)
    rows = pd.concat([template] * 6, ignore_index=True)
    for position in range(6):
        rows.loc[position, config.REPORT_REFERENCE_COLUMN] = "SC09" + str(90001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
    depth, width, water = "defect_depth_mm", "defect_width_mm", "water_depth_mm"
    reach, clear = config.WATER_REACH_COLUMN, config.CLEAR_HEIGHT_COLUMN
    lean, height = config.COLUMN_LEAN_COLUMN, config.COLUMN_HEIGHT_COLUMN
    planned, previous = config.PLANNED_CREW_HOURS_COLUMN, "previous_reports_at_location"
    impossible = [(0, depth, "0"), (0, width, "-5"), (0, water, "0 cm"), (0, reach, "0"),
                  (0, clear, "-1 m"), (0, height, "0 ft"), (0, lean, "-0.5"), (0, planned, "0"),
                  (0, previous, "-1"), (1, depth, "0.01 in"), (1, width, "0.4"),
                  (1, water, "-0.2 in"), (1, reach, "0.1 ft"), (1, clear, "0.04"),
                  (1, lean, "90.1 deg"), (1, planned, "-30 min")]
    for position, column_name, written in impossible:
        rows.loc[position, column_name] = written
    possible = [(2, depth, "1", 1), (2, width, "0.02 in", 1), (2, water, "0.1 cm", 1),
                (2, reach, "0.1", 0.1), (2, clear, "0.2 ft", 0.1), (2, height, "0.1 m", 0.1),
                (2, lean, "90", 90.0), (2, planned, "0.1", 0.1), (2, previous, "0", 0),
                (5, lean, "0 deg", 0.0), (5, planned, "3 min", 0.1), (5, reach, "0.17 ft", 0.1)]
    for position, column_name, written, _wanted in possible:
        rows.loc[position, column_name] = written
    blanked = ["location_fixed", depth, config.STREET_CATEGORY_COLUMN,
               config.LIGHTING_FAULT_COLUMN]
    for position, blank_tokens in [(3, [" N/A ", "not known", "Caller unsure", "NULL"]),
                                   (4, ["-", "NA", " n/a", ""])]:
        for column_name, blank_token in zip(blanked, blank_tokens, strict=True):
            rows.loc[position, column_name] = blank_token
    planted_path = str(tmp_path / "planted_impossible.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.REPORT_REFERENCE_COLUMN}.union(
        [column_name for _position, column_name, _written in impossible], blanked)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for six planted reports, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.REPORT_REFERENCE_COLUMN)
    planted_references = ["SC09" + str(90001 + position) for position in range(6)]
    assert sorted(cleaned.index) == planted_references, (
        "Six planted reports cleaned to the rows "
        + str(sorted(cleaned.index))
        + ". A reading that cannot be true, or an answer left blank, never takes its row with "
        "it."
    )
    still_read = [column_name + " " + written for position, column_name, written in impossible
                  if not pd.isna(cleaned.loc[planted_references[position], column_name])]
    assert len(still_read) == 0, (
        "Readings that cannot be true were kept: "
        + str(still_read)
        + ". A depth, a width or standing water of nought or less, a reach, a clear height or a "
        "column height of nought or less, a lean below nought or above 90 degrees, planned crew "
        "hours of nought or less and a negative count of previous reports cannot be true, so the "
        "reading is made blank, never corrected, and the row stays - judged on the reading as "
        "the rules read it, so 0.01 in, 0.25 mm, is nought millimetres."
    )
    lost = [column_name + " " + written + " read as "
            + str(cleaned.loc[planted_references[position], column_name])
            for position, column_name, written, wanted in possible
            if pd.isna(cleaned.loc[planted_references[position], column_name])
            or cleaned.loc[planted_references[position], column_name] != wanted]
    assert len(lost) == 0, (
        "Readings at the edge of what can be true were not kept as the rules read them: "
        + str(lost)
        + ". A depth, a width or standing water of one millimetre, a reach, a clear height or a "
        "column height of 0.1 metres, a lean of nought or of 90 degrees, planned crew hours of "
        "0.1 and no previous reports are readings like any other, and 0.02 in reads as one "
        "millimetre and 3 min as 0.1 hours."
    )
    unblanked = [str(position) + " " + column_name for position in [3, 4]
                 for column_name in blanked
                 if not pd.isna(cleaned.loc[planted_references[position], column_name])]
    assert len(unblanked) == 0, (
        "An answer written N/A, not known, caller unsure, null, - or NA, with spaces around it "
        "or in any capitals, or left empty, was read as a value on planted rows "
        + str(unblanked)
        + ". Each channel's way of leaving an answer blank is a blank answer."
    )
    raw_logs = raw_frames[config.RAW_REPAIR_LOGS_EVAL_FILE]
    log_template = raw_logs.drop_duplicates(
        config.REPORT_REFERENCE_COLUMN).head(4).reset_index(drop=True).copy()
    log_rows = log_template.copy()
    log_rows[config.REPORT_REFERENCE_COLUMN] = ["SC0991001", "SC0991002", "SC0991003",
                                                "SC0991004"]
    log_rows[config.LOG_TARGET_COLUMN] = ["-2", "0", "2.5 hrs", "6 min"]
    log_rows.loc[2, planned] = "-3 hrs"
    log_rows.loc[3, "crew_size"] = "0"
    log_rows = pd.concat([log_rows, log_rows.iloc[[2]]], ignore_index=True)
    log_path = str(tmp_path / "planted_logs.csv")
    log_rows.to_csv(log_path, index=False)
    try:
        cleaned_logs = clean_repair_log_dataset(load_raw_dataset(log_path))
    except Exception as error:
        cleaned_logs = error
    assert isinstance(cleaned_logs, pd.DataFrame) and {
            config.REPORT_REFERENCE_COLUMN, planned, "crew_size",
            config.LOG_TARGET_COLUMN} <= set(cleaned_logs.columns), (
        "clean_repair_log_dataset gave "
        + repr(cleaned_logs)[:200]
        + " for five planted repair logs, where it returns the cleaned table."
    )
    kept = cleaned_logs[config.REPORT_REFERENCE_COLUMN].astype(str).tolist()
    actual_hours = cleaned_logs[config.LOG_TARGET_COLUMN].tolist()
    assert (kept == ["SC0991003", "SC0991004"] and actual_hours == [2.5, 0.1]
            and pd.isna(cleaned_logs[planned].iloc[0])
            and pd.isna(cleaned_logs["crew_size"].iloc[1])), (
        "Five planted repair logs, the third sent twice exactly, cleaned to "
        + str(cleaned_logs[[config.REPORT_REFERENCE_COLUMN, planned, "crew_size",
                            config.LOG_TARGET_COLUMN]].to_dict("records"))
        + ". Actual crew hours of -2 or of nought cannot be true and are made blank, and being "
        "the answer the log cannot be learned from, so the row goes; six minutes are 0.1 hours "
        "and stay; negative planned crew hours and a crew of nobody are made blank and the log "
        "stays; and a log sent twice exactly is one log."
    )
    raw_sentences = raw_frames[config.RAW_STATEMENT_SENTENCES_FILE]
    raw_register = load_raw_dataset(config.STREET_WORKS_REGISTER_FILE)
    history_keys = pd.concat([template] * 2, ignore_index=True)
    history_keys[config.REPORT_REFERENCE_COLUMN] = "SC0992001"
    history_keys[config.REVISION_NUMBER_COLUMN] = ["1", "1"]
    history_keys[planned] = ["1.5", "2.5"]
    blank_revision = template.copy()
    blank_revision[config.REPORT_REFERENCE_COLUMN] = "SC0992002"
    blank_revision[config.REVISION_NUMBER_COLUMN] = "N/A"
    shared_logs = log_template.head(2).copy()
    shared_logs[config.REPORT_REFERENCE_COLUMN] = ["SC-992003", "992003"]
    unnamed_logs = log_template.head(2).copy()
    unnamed_logs.loc[unnamed_logs.index[0], config.REPORT_REFERENCE_COLUMN] = "-"
    unnamed_sentences = raw_sentences.head(2).copy()
    unnamed_sentences[config.SENTENCE_ID_COLUMN] = ["", "SEN-9920041"]
    shared_sentences = raw_sentences.head(2).copy()
    shared_sentences[config.SENTENCE_ID_COLUMN] = "SEN-9920042"
    different_permits = raw_register.drop_duplicates().head(2)
    shared_register = different_permits.copy()
    shared_register[config.PERMIT_REFERENCE_COLUMN] = "SWP0051207"
    unstated_register = different_permits.copy()
    unstated_register.loc[unstated_register.index[1], config.PERMIT_STATUS_COLUMN] = None
    for label, cleaner, frame, named in [
            ("two rows of the same report at the same revision", clean_history_dataset,
             history_keys, ["SC0992001", config.REVISION_NUMBER_COLUMN]),
            ("a report whose revision is left blank", clean_history_dataset,
             blank_revision, ["SC0992002", config.REVISION_NUMBER_COLUMN]),
            ("two repair logs of one report", clean_repair_log_dataset,
             shared_logs, ["SC0992003", config.REPORT_REFERENCE_COLUMN]),
            ("a repair log with no report reference", clean_repair_log_dataset,
             unnamed_logs, [config.REPORT_REFERENCE_COLUMN]),
            ("a sentence with no identifier", clean_sentence_dataset, unnamed_sentences,
             [config.SENTENCE_ID_COLUMN]),
            ("two sentences sharing an identifier", clean_sentence_dataset, shared_sentences,
             ["SEN-9920042", config.SENTENCE_ID_COLUMN]),
            ("two different permits of the register both written SWP0051207",
             clean_register_records, shared_register,
             ["SWP0051207", config.PERMIT_REFERENCE_COLUMN]),
            ("a permit of the register with no status", clean_register_records,
             unstated_register, [config.PERMIT_STATUS_COLUMN])]:
        keyed_path = str(tmp_path / "keyed.csv")
        frame.to_csv(keyed_path, index=False)
        refusal = None
        try:
            cleaner(load_raw_dataset(keyed_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and all(word in refusal for word in named), (
            "Cleaning "
            + label
            + " gave "
            + str(refusal)
            + ". Every report revision, repair log, sentence and permit of the register is named "
            "and appears once: a blank or repeated key is refused with a ValueError naming "
            + " and ".join(named)
            + ", never dropped or numbered quietly."
        )


def test_schemas_leave_out_outcome_and_identifiers(saved_models):
    from config import config

    assert saved_models["problem"] is None, saved_models["problem"]
    repeat_schema = saved_models["repeat_schema"]
    crew_hours_schema = saved_models["crew_hours_schema"]
    for schema_name, schema, configured in [
            ("repeat", repeat_schema,
             config.REPEAT_NUMERIC_FEATURES + config.REPEAT_CATEGORICAL_FEATURES),
            ("crew hours", crew_hours_schema,
             config.CREW_HOURS_NUMERIC_FEATURES + config.CREW_HOURS_CATEGORICAL_FEATURES)]:
        assert len(schema) == len(set(schema)) and sorted(schema) == sorted(configured), (
            "The saved "
            + schema_name
            + " schema names "
            + str(schema)
            + " where the configuration's features are "
            + str(configured)
            + ", each once."
        )
    repeat_never = [config.HISTORY_TARGET_COLUMN, "review_outcome", "repeat_cause",
                    config.REPORT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
                    config.LOCATION_DESCRIPTION_COLUMN]
    crew_hours_never = [config.LOG_TARGET_COLUMN, "completion_code", "repair_method",
                        "traffic_management", "crew_size", config.HISTORY_TARGET_COLUMN,
                        config.REPORT_REFERENCE_COLUMN]
    for schema_name, schema, never in [("repeat", repeat_schema, repeat_never),
                                       ("crew hours", crew_hours_schema, crew_hours_never)]:
        leaked = sorted(set(schema) & set(never))
        assert len(leaked) == 0, (
            "The saved "
            + schema_name
            + " schema feeds the model "
            + str(leaked)
            + ". The answer the model learns, an identifier, and anything recorded only once "
            "the decision was taken or the repair was done are never features: a model fed them "
            "learns what it cannot know when the report is reviewed, or learns which report it is "
            "rather than what was reported."
        )
    counts_used = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
                   if column_name in repeat_schema]
    flag_columns = list(config.COORDINATION_TRIGGER_FLAG_COLUMNS.values())
    flags_used = {schema_name: [column_name for column_name in flag_columns
                                if column_name in schema]
                  for schema_name, schema in [("repeat", repeat_schema),
                                              ("crew hours", crew_hours_schema)]}
    assert (counts_used == list(config.REQUIREMENT_COUNT_RESULTS)
            and flags_used == {"repeat": flag_columns, "crew hours": flag_columns}), (
        "The repeat schema uses the requirement counts "
        + str(counts_used)
        + ", and the two schemas use the coordination trigger flags "
        + str(flags_used)
        + ", where the repeat model is fed all three counts - requirements Met, Not met and Not "
        "documented - and both models every coordination trigger's flag, so they learn from the "
        "screening a live report is shown with."
    )


def test_counts_and_triggers_use_own_standards(processed_frames, defect_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_coordination_trigger_flags, add_requirement_counts
    from tools.screening_tools import find_coordination_triggers, screen_report

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    standards = defect_standards["standards"]
    history = processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE]
    sample = history.groupby(config.DEFECT_TYPE_COLUMN, group_keys=False).head(40)
    sample = sample.reset_index(drop=True)
    linked_rows = int((sample[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT).sum())
    assert linked_rows > 0, (
        "The first past reports of each defect type hold no report with a linked defect type, "
        "where the cleaned report history holds many."
    )
    try:
        counted = add_requirement_counts(sample, standards)
    except Exception as error:
        counted = error
    assert isinstance(counted, pd.DataFrame), (
        "add_requirement_counts gave "
        + repr(counted)[:200]
        + ", where it returns the reports it was given with their requirement counts beside "
        "them."
    )
    missing = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
               if column_name not in counted.columns]
    assert len(missing) == 0 and len(counted) == len(sample), (
        "add_requirement_counts returned "
        + str(len(counted))
        + " rows of the "
        + str(len(sample))
        + " it was given, without the columns "
        + str(missing)
        + "."
    )
    assert config.REPORT_REFERENCE_COLUMN in counted and list(
            counted[config.REPORT_REFERENCE_COLUMN]) == list(
            sample[config.REPORT_REFERENCE_COLUMN]), (
        "add_requirement_counts returned the reports in another order, so each row's counts no "
        "longer sit beside its own answers."
    )
    try:
        flagged = add_coordination_trigger_flags(counted)
    except Exception as error:
        flagged = error
    flag_columns = list(config.COORDINATION_TRIGGER_FLAG_COLUMNS.values())
    assert isinstance(flagged, pd.DataFrame) and set(flag_columns) <= set(flagged.columns) and list(
            flagged[config.REPORT_REFERENCE_COLUMN]) == list(
            sample[config.REPORT_REFERENCE_COLUMN]), (
        "add_coordination_trigger_flags gave "
        + repr(flagged)[:200]
        + ", where it returns the reports it was given, in their order, with a flag for each "
        "coordination trigger "
        + str(flag_columns)
        + " beside them."
    )
    wrong = []
    for report, flagged_row in zip(sample.to_dict("records"), flagged.to_dict("records"),
                                   strict=True):
        screening = screen_report(report, standards)
        assert isinstance(screening, dict) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
                screening), (
            "screen_report returned "
            + str(screening)[:200]
            + " for "
            + report[config.REPORT_REFERENCE_COLUMN]
            + ", where it returns the screening with its counts "
            + str(list(config.REQUIREMENT_COUNT_RESULTS))
            + "."
        )
        own_count = len(standards[report[config.DEFECT_TYPE_COLUMN]]["requirements"])
        if report[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT:
            own_count += len(standards[report[config.LINKED_DEFECT_TYPE_COLUMN]]["requirements"])
        expected = [screening[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        got = [flagged_row[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        if got != expected or sum(got) != own_count:
            wrong.append(report[config.REPORT_REFERENCE_COLUMN] + " counted " + str(got)
                         + " where its standards' screening gives " + str(expected)
                         + " of " + str(own_count) + " requirements")
        triggers = find_coordination_triggers(report)
        expected_flags = [int(trigger_name in triggers)
                          for trigger_name in config.COORDINATION_TRIGGER_FLAG_COLUMNS]
        got_flags = [flagged_row[column_name] for column_name in flag_columns]
        if got_flags != expected_flags:
            wrong.append(report[config.REPORT_REFERENCE_COLUMN] + " flagged " + str(got_flags)
                         + " where its coordination triggers " + str(triggers) + " give "
                         + str(expected_flags))
    assert len(wrong) == 0, (
        "The requirement counts and the trigger flags disagree with the screening under each "
        "report's own standards: "
        + "; ".join(wrong[:4])
        + ". Each past report is screened against the standard of its own defect type and of "
        "its linked type, as a live report is; its requirements Met, Not met and Not documented "
        "add up to both standards' requirements, and each coordination trigger it meets is "
        "flagged 1, the others 0."
    )
    logs = processed_frames[config.CLEAN_REPAIR_LOGS_EVAL_FILE]
    log_sample = logs.groupby(config.DEFECT_TYPE_COLUMN, group_keys=False).head(30).reset_index(
        drop=True)
    try:
        flagged_logs = add_coordination_trigger_flags(log_sample)
    except Exception as error:
        flagged_logs = error
    assert isinstance(flagged_logs, pd.DataFrame) and set(flag_columns) <= set(
            flagged_logs.columns), (
        "add_coordination_trigger_flags gave "
        + repr(flagged_logs)[:200]
        + " for repair logs, where the crew hours features flag the same coordination triggers."
    )
    log_wrong = [log[config.REPORT_REFERENCE_COLUMN]
                 for log, flagged_log in zip(log_sample.to_dict("records"),
                                             flagged_logs.to_dict("records"), strict=True)
                 if [flagged_log[column_name] for column_name in flag_columns]
                 != [int(trigger_name in find_coordination_triggers(log))
                     for trigger_name in config.COORDINATION_TRIGGER_FLAG_COLUMNS]]
    assert len(log_wrong) == 0, (
        "The repair logs "
        + str(log_wrong[:4])
        + " are flagged otherwise than their own coordination triggers. A repair log is flagged "
        "the way a past report is."
    )
    crossed = dict(standards)
    crossed[config.CARRIAGEWAY] = standards[config.FOOTWAY]
    carriageway = sample[(sample[config.DEFECT_TYPE_COLUMN] == config.CARRIAGEWAY)].head(1)
    refusal = None
    try:
        add_requirement_counts(carriageway, crossed)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    footway_document = config.DEFECT_STANDARD_DOCUMENTS[config.FOOTWAY]
    assert (refusal is not None and config.CARRIAGEWAY in refusal
            and footway_document in refusal), (
        "A carriageway report screened with the footway standard handed over under its defect "
        "type gave "
        + str(refusal)
        + ". The two standards are twins written from one template, and a standard written for "
        "another defect type is refused, naming the defect type and the document, never used to "
        "screen it."
    )
    linked_to_drainage = carriageway.copy()
    linked_to_drainage[config.LINKED_DEFECT_TYPE_COLUMN] = config.DRAINAGE
    without_standard = dict(standards)
    del without_standard[config.DRAINAGE]
    for meaning, reports in [
            ("a drainage report",
             sample[sample[config.DEFECT_TYPE_COLUMN] == config.DRAINAGE].head(1)),
            ("a carriageway report linked to drainage", linked_to_drainage)]:
        refusal = None
        try:
            add_requirement_counts(reports, without_standard)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert (refusal is not None and config.DRAINAGE in refusal
                and reports[config.REPORT_REFERENCE_COLUMN].iloc[0] in refusal), (
            meaning[0].upper()
            + meaning[1:]
            + " with no drainage standard handed over gave "
            + str(refusal)
            + ". A report whose defect type or linked type has no standard cannot be screened "
            "and is refused, naming the report and the defect type."
        )


def test_inputs_follow_schema_order_with_gaps(processed_frames, saved_models, defect_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_coordination_trigger_flags,
                                               add_requirement_counts, build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    try:
        history = add_coordination_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE].head(60),
            defect_standards["standards"]))
        logs = add_coordination_trigger_flags(
            processed_frames[config.CLEAN_REPAIR_LOGS_EVAL_FILE].head(60))
    except Exception as error:
        history = logs = error
    repeat_columns = config.REPEAT_NUMERIC_FEATURES + config.REPEAT_CATEGORICAL_FEATURES
    crew_hours_columns = (config.CREW_HOURS_NUMERIC_FEATURES
                          + config.CREW_HOURS_CATEGORICAL_FEATURES)
    assert (isinstance(history, pd.DataFrame) and set(repeat_columns) <= set(history.columns)
            and isinstance(logs, pd.DataFrame) and set(crew_hours_columns) <= set(logs.columns)), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the reports with every repeat feature "
        + str(repeat_columns)
        + " beside them, and the repair logs with every crew hours feature."
    )
    for name, cleaned, numeric_columns, categorical_columns, schema in [
            ("repeat", history, config.REPEAT_NUMERIC_FEATURES,
             config.REPEAT_CATEGORICAL_FEATURES, saved_models["repeat_schema"]),
            ("crew hours", logs, config.CREW_HOURS_NUMERIC_FEATURES,
             config.CREW_HOURS_CATEGORICAL_FEATURES, saved_models["crew_hours_schema"])]:
        frame = cleaned.copy()
        for column_name in numeric_columns:
            frame[column_name] = frame[column_name].astype(float)
            frame.loc[frame.index[0], column_name] = float("nan")
        for column_name in categorical_columns:
            frame[column_name] = frame[column_name].astype(object)
            frame.loc[frame.index[1], column_name] = None
        shuffled = frame[list(reversed(list(frame.columns)))]
        features = build_feature_frame(shuffled, numeric_columns, categorical_columns)
        assert isinstance(features, pd.DataFrame), (
            "build_feature_frame returned "
            + type(features).__name__
            + " for the "
            + name
            + " inputs, where it returns a table of the model's inputs."
        )
        assert list(features.columns) == schema, (
            "The "
            + name
            + " inputs are built in the order "
            + str(list(features.columns))
            + " where the saved schema reads "
            + str(schema)
            + ". The model takes its inputs in the order it was trained on, whatever order the "
            "table it is given holds them in."
        )
        assert len(features) == len(frame) and list(features.index) == list(frame.index), (
            "build_feature_frame returned "
            + str(len(features))
            + " rows for "
            + str(len(frame))
            + ", or in another order."
        )
        for column_name in numeric_columns:
            same = np.allclose(features[column_name].to_numpy(dtype=float),
                               frame[column_name].to_numpy(dtype=float), equal_nan=True)
            assert same, (
                column_name
                + " is changed on its way into the "
                + name
                + " model. A number goes in as it was cleaned, a gap as a gap."
            )
        for column_name in categorical_columns:
            gaps = frame[column_name].isna()
            built = features[column_name]
            gaps_as_nan = all(isinstance(value, float) and np.isnan(value)
                              for value in built[gaps])
            assert gaps_as_nan and list(built[~gaps]) == list(frame.loc[~gaps, column_name]), (
                column_name
                + " is changed on its way into the "
                + name
                + " model: a gap became "
                + str(sorted({repr(value) for value in built[gaps]})[:3])
                + ". An answer in words goes in as cleaned and a gap as NaN, which the model's "
                "own imputer fills, never None, the text 'nan' or an empty string."
            )


def test_models_fit_schema_and_fill_gaps(processed_frames, saved_models, defect_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_coordination_trigger_flags,
                                               add_requirement_counts, build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    for name, label, schema in [("repeat", "repeat", saved_models["repeat_schema"]),
                                ("crew_hours", "crew hours", saved_models["crew_hours_schema"])]:
        fitted_on = list(getattr(saved_models[name], "feature_names_in_", []))
        assert fitted_on == schema, (
            "The saved "
            + label
            + " model was fitted on "
            + str(fitted_on)
            + " while its schema names "
            + str(schema)
            + ". A model saved beside a schema it was not fitted on scores live reports on the "
            "wrong columns."
        )
    planned, category = config.PLANNED_CREW_HOURS_COLUMN, config.STREET_CATEGORY_COLUMN
    history_train = processed_frames[config.CLEAN_REPORT_HISTORY_TRAIN_FILE]
    try:
        history = add_coordination_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE].dropna(
                subset=[planned, category]).head(25),
            defect_standards["standards"]))
    except Exception as error:
        history = error
    assert isinstance(history, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the reports with them beside their answers."
    )
    features = build_feature_frame(history, config.REPEAT_NUMERIC_FEATURES,
                                   config.REPEAT_CATEGORICAL_FEATURES)
    log_features = build_feature_frame(
        add_coordination_trigger_flags(processed_frames[config.CLEAN_REPAIR_LOGS_EVAL_FILE].dropna(
            subset=[planned, category]).head(25)),
        config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and isinstance(log_features, pd.DataFrame), (
        "build_feature_frame returned "
        + type(features).__name__
        + " and "
        + type(log_features).__name__
        + " for the repeat and crew hours inputs, where it returns a table of the model's "
        "inputs."
    )
    with_gaps = features.copy()
    with_gaps[planned] = np.nan
    with_gaps[category] = np.nan
    filled = features.copy()
    filled[planned] = float(history_train[planned].median())
    filled[category] = history_train[category].mode().iloc[0]
    gap_scores = saved_models["repeat"].predict_proba(with_gaps)[:, 1]
    filled_scores = saved_models["repeat"].predict_proba(filled)[:, 1]
    assert np.isfinite(gap_scores).all() and np.allclose(gap_scores, filled_scores), (
        "A report with its planned crew hours and its street category left blank scores "
        + str(np.round(gap_scores[:3], 4))
        + " where the same report with the training median and the commonest street category "
        "scores "
        + str(np.round(filled_scores[:3], 4))
        + ". The pipeline fills a blank number with the training data's "
        + config.NUMBER_FILL_STRATEGY
        + " and a blank answer in words with its "
        + config.CATEGORY_FILL_STRATEGY
        + " value, so a live report with a gap is still scored."
    )
    logs_train = processed_frames[config.CLEAN_REPAIR_LOGS_TRAIN_FILE]
    log_gaps = log_features.copy()
    log_gaps[planned] = np.nan
    log_gaps[category] = np.nan
    log_filled = log_features.copy()
    log_filled[planned] = float(logs_train[planned].median())
    log_filled[category] = logs_train[category].mode().iloc[0]
    gap_hours = saved_models["crew_hours"].predict(log_gaps)
    filled_hours = saved_models["crew_hours"].predict(log_filled)
    assert np.isfinite(gap_hours).all() and np.allclose(gap_hours, filled_hours), (
        "A repair with its planned crew hours and its street category left blank is expected to "
        "take "
        + str(np.round(gap_hours[:3], 1))
        + " crew hours where the same repair filled as training filled it gives "
        + str(np.round(filled_hours[:3], 1))
        + "."
    )


def test_quality_report_accounts_for_rows(raw_frames, processed_frames):
    import json
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.quality_profiler import profile_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert os.path.isfile(config.DATA_QUALITY_REPORT_FILE), (
        "There is no quality report at "
        + config.DATA_QUALITY_REPORT_FILE
        + ". The data pipeline writes it when it cleans the five extracts."
    )
    with open(config.DATA_QUALITY_REPORT_FILE, encoding="utf-8") as report_file:
        report = json.load(report_file)
    datasets = [
        ("report_history_train", config.RAW_REPORT_HISTORY_TRAIN_FILE,
         config.CLEAN_REPORT_HISTORY_TRAIN_FILE, config.HISTORY_TARGET_COLUMN),
        ("report_history_eval", config.RAW_REPORT_HISTORY_EVAL_FILE,
         config.CLEAN_REPORT_HISTORY_EVAL_FILE, config.HISTORY_TARGET_COLUMN),
        ("repair_logs_train", config.RAW_REPAIR_LOGS_TRAIN_FILE,
         config.CLEAN_REPAIR_LOGS_TRAIN_FILE, config.LOG_TARGET_COLUMN),
        ("repair_logs_eval", config.RAW_REPAIR_LOGS_EVAL_FILE, config.CLEAN_REPAIR_LOGS_EVAL_FILE,
         config.LOG_TARGET_COLUMN),
        ("statement_sentences", config.RAW_STATEMENT_SENTENCES_FILE,
         config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_TARGET_COLUMN)]
    assert isinstance(report, dict) and sorted(report) == sorted(
            dataset[0] for dataset in datasets) and all(
            isinstance(profile, dict) for profile in report.values()), (
        config.DATA_QUALITY_REPORT_FILE
        + " profiles "
        + str(sorted(report))[:300]
        + " where it profiles each of the five extracts under its own name."
    )
    reasons = ["duplicate_rows", "rows_missing_target", "answered_earlier_revisions_removed",
               "rows_with_impossible_target", "rows_with_too_few_words"]
    counted_keys = ["rows_received", "rows_kept", "rows_removed", "duplicate_rows",
                    "rows_missing_target", "columns"]
    for name, raw_path, clean_path, target_column in datasets:
        profile = report[name]
        try:
            profiled = profile_dataset(load_raw_dataset(raw_path), processed_frames[clean_path],
                                       target_column)
        except Exception as error:
            profiled = error
        assert isinstance(profiled, dict), (
            "profile_dataset gave "
            + repr(profiled)[:200]
            + " for "
            + name
            + ", where it returns the counts that profile the extract."
        )
        expected = {key: profiled.get(key) for key in counted_keys}
        expected.update({
            "rows_received": len(raw_frames[raw_path]),
            "rows_kept": len(processed_frames[clean_path]),
            "rows_removed": len(raw_frames[raw_path]) - len(processed_frames[clean_path]),
            "columns": int(processed_frames[clean_path].shape[1]),
        })
        got = {key: profile.get(key) for key in counted_keys}
        assert got == expected, (
            "The quality report gives "
            + name
            + " as "
            + str(got)
            + " where the files and profile_dataset give "
            + str(expected)
            + ". Rows received are the raw rows and rows kept the cleaned rows, and the report "
            "holds what profile_dataset counts in the extract as the pipeline read it."
        )
        explained = sum(int(profile.get(reason, 0)) for reason in reasons)
        assert explained == expected["rows_removed"], (
            "The quality report's reasons for "
            + name
            + " add up to "
            + str(explained)
            + " rows where "
            + str(expected["rows_removed"])
            + " were removed: "
            + str({reason: profile.get(reason) for reason in reasons if reason in profile})
            + ". Every removed row is a repeat, a row with no answer, an earlier revision, an "
            "answer that cannot be true or a sentence too short to read."
        )
    planted_raw = pd.DataFrame({
        config.REPORT_REFERENCE_COLUMN: ["SC0960001", "SC0960001", "SC0960002", "SC0960002",
                                         "SC0960003", "SC0960004"],
        config.STREET_NAME_COLUMN: ["Mill Lane", "Mill Lane", "Northgate", "Northgate",
                                    "Kiln Lane", "Ferry Lane"],
        config.HISTORY_TARGET_COLUMN: [config.BOOLEAN_NO, config.BOOLEAN_NO, None, None, None,
                                       config.BOOLEAN_YES]})
    planted_clean = planted_raw.iloc[[0, 5]].reset_index(drop=True)
    try:
        profiled = profile_dataset(planted_raw, planted_clean, config.HISTORY_TARGET_COLUMN)
    except Exception as error:
        profiled = error
    expected = {"rows_received": 6, "rows_kept": 2, "rows_removed": 4, "duplicate_rows": 2,
                "rows_missing_target": 2, "columns": 3}
    got = ({key: profiled.get(key) for key in expected} if isinstance(profiled, dict)
           else profiled)
    assert got == expected, (
        "Six planted rows - a report sent twice exactly, a report with no reported again answer "
        "sent twice exactly, another report with no answer and one answered - profiled against "
        "the two rows kept as "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A repeat is counted once, beside its first copy, and a row missing its answer is "
        "counted among the first copies only."
    )


def test_repeat_model_auc_and_calibration(processed_frames, saved_models, defect_standards):
    import copy

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from config import config
    from data_pipeline.feature_builder import (add_coordination_trigger_flags,
                                               add_requirement_counts, build_feature_frame)
    from modeling.model_evaluation import evaluate_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    standards = defect_standards["standards"]
    history_train = processed_frames[config.CLEAN_REPORT_HISTORY_TRAIN_FILE]
    try:
        held_out = add_coordination_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE], standards))
        trained_on = add_coordination_trigger_flags(add_requirement_counts(
            history_train[history_train[config.DEFECT_TYPE_COLUMN]
                          == config.VEGETATION].head(3000), standards))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.HISTORY_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the reports with their reported again answers, requirement counts "
        "and trigger flags."
    )
    held_out_features = build_feature_frame(held_out, config.REPEAT_NUMERIC_FEATURES,
                                            config.REPEAT_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.REPEAT_NUMERIC_FEATURES,
                                         config.REPEAT_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    held_out_answers = held_out[config.HISTORY_TARGET_COLUMN].map(
        config.REPEAT_TARGET_ENCODING)
    train_answers = trained_on[config.HISTORY_TARGET_COLUMN].map(
        config.REPEAT_TARGET_ENCODING)
    model = saved_models["repeat"]
    scores = model.predict_proba(held_out_features)[:, 1]
    measured = roc_auc_score(held_out_answers, scores)
    assert measured >= config.MINIMUM_REPEAT_ROC_AUC, (
        "The repeat model ranks the held-out reports with a ROC AUC of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_REPEAT_ROC_AUC)
        + " the configuration sets. It is scored on the later reports, which it never saw, "
        "with the requirement counts worked out under each report's own standards."
    )
    assert abs(float(scores.mean()) - float(held_out_answers.mean())) <= 0.03, (
        "The repeat model's scores average "
        + str(round(float(scores.mean()), 4))
        + " on the held-out reports, where "
        + str(round(float(held_out_answers.mean()), 4))
        + " of their defects were reported again. A repeat likelihood is read as the share of "
        "defects like this one that were reported again, so the scores average the share that "
        "were."
    )
    evaluation = evaluate_classifier(copy.deepcopy(model), train_features, train_answers,
                                     held_out_features, held_out_answers)
    missing = sorted({"train_roc_auc", "held_out_roc_auc", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the repeat model's scores cannot be read."
    )
    train_measured = roc_auc_score(train_answers, model.predict_proba(train_features)[:, 1])
    assert (abs(evaluation["held_out_roc_auc"] - measured) < 0.006
            and abs(evaluation["train_roc_auc"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_roc_auc"] - evaluation["held_out_roc_auc"])) < 0.011), (
        "Given the held-out reports and the first training reports of one defect type, the "
        "evaluation reports a held-out ROC AUC of "
        + str(evaluation["held_out_roc_auc"])
        + ", a training ROC AUC of "
        + str(evaluation["train_roc_auc"])
        + " and a gap of "
        + str(evaluation["generalisation_gap"])
        + ", where the model's own scores on those rows give "
        + str(round(measured, 4))
        + " and "
        + str(round(train_measured, 4))
        + ". evaluate_classifier scores the fitted model it is given on the rows it is given, "
        "without fitting it again, and the gap is the training score less the held-out one."
    )


def test_crew_hours_model_r2_and_floor(processed_frames, saved_models):
    import copy

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    from config import config
    from data_pipeline.feature_builder import add_coordination_trigger_flags, build_feature_frame
    from modeling.model_evaluation import evaluate_regressor

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    training_logs = processed_frames[config.CLEAN_REPAIR_LOGS_TRAIN_FILE]
    try:
        held_out = add_coordination_trigger_flags(
            processed_frames[config.CLEAN_REPAIR_LOGS_EVAL_FILE])
        trained_on = add_coordination_trigger_flags(
            training_logs[training_logs[config.DEFECT_TYPE_COLUMN]
                          == config.FOOTWAY].head(3000))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.LOG_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "add_coordination_trigger_flags gave "
        + repr(held_out)[:200]
        + ", where it returns the repair logs with their trigger flags beside them."
    )
    held_out_features = build_feature_frame(held_out, config.CREW_HOURS_NUMERIC_FEATURES,
                                            config.CREW_HOURS_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.CREW_HOURS_NUMERIC_FEATURES,
                                         config.CREW_HOURS_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    model = saved_models["crew_hours"]
    forecast = model.predict(held_out_features)
    measured = r2_score(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert measured >= config.MINIMUM_CREW_HOURS_R2, (
        "The crew hours model explains the held-out actual crew hours with an R2 of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_CREW_HOURS_R2)
        + " the configuration sets."
    )
    floor = float(training_logs[config.LOG_TARGET_COLUMN].min())
    assert float(forecast.min()) >= floor, (
        "The crew hours model expects a held-out repair to take "
        + str(round(float(forecast.min()), 1))
        + " crew hours, fewer than the "
        + str(floor)
        + " of the quickest repair it learned from. No repair was ever done in fewer, so a "
        "shorter forecast is the model inventing one."
    )
    training_average = float(training_logs[config.LOG_TARGET_COLUMN].mean())
    average_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN],
                                        [training_average] * len(held_out))
    model_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert model_error < 0.75 * average_error, (
        "The crew hours model is off by "
        + str(round(model_error, 1))
        + " crew hours on average, against "
        + str(round(average_error, 1))
        + " for the average repair's hours given to every repair. A forecast that does little "
        "better than the average tells the Highways Inspector nothing about this report."
    )
    evaluation = evaluate_regressor(copy.deepcopy(model), train_features,
                                    trained_on[config.LOG_TARGET_COLUMN], held_out_features,
                                    held_out[config.LOG_TARGET_COLUMN])
    missing = sorted({"train_r2", "held_out_r2", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_regressor returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the crew hours model's scores cannot be read."
    )
    train_measured = r2_score(trained_on[config.LOG_TARGET_COLUMN], model.predict(train_features))
    assert (abs(evaluation["held_out_r2"] - measured) < 0.006
            and abs(evaluation["train_r2"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_r2"] - evaluation["held_out_r2"])) < 0.011), (
        "Given the held-out logs and the first training logs of one defect type, "
        "evaluate_regressor reports "
        + str({key: evaluation.get(key) for key in ["train_r2", "held_out_r2",
                                                    "generalisation_gap"]})
        + " where the model's own forecasts on those rows give a held-out R2 of "
        + str(round(measured, 4))
        + " and a training R2 of "
        + str(round(train_measured, 4))
        + ". It scores the fitted model it is given on the rows it is given, without fitting it "
        "again."
    )


def test_repeat_rises_with_failures_and_conflicts(processed_frames, saved_models,
                                                  defect_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_coordination_trigger_flags,
                                               add_requirement_counts, build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    try:
        held_out = add_coordination_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE],
            defect_standards["standards"]))
    except Exception as error:
        held_out = error
    assert isinstance(held_out, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the reports with them beside their answers."
    )
    features = build_feature_frame(held_out, config.REPEAT_NUMERIC_FEATURES,
                                   config.REPEAT_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
            features.columns), (
        "build_feature_frame returned "
        + type(features).__name__
        + ", where it returns the model's inputs with the requirement counts among them."
    )
    model = saved_models["repeat"]
    met_column = config.REQUIREMENTS_MET_COUNT_COLUMN
    before = features[features[met_column] >= 1]
    assert len(before) > 50, (
        "Only "
        + str(len(before))
        + " held-out reports are counted with a requirement Met, where the screening finds far "
        "more."
    )
    after = before.copy()
    after[met_column] = after[met_column] - 1
    after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] = (
        after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] + 1)
    before_scores = model.predict_proba(before)[:, 1]
    after_scores = model.predict_proba(after)[:, 1]
    rising = float((after_scores > before_scores).mean())
    assert rising >= 0.95 and (after_scores.mean() - before_scores.mean()) >= 0.03, (
        "Turning one requirement from Met to Not met raises the repeat score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(before))
        + " held-out reports it applies to, and moves the average from "
        + str(round(float(before_scores.mean()), 4))
        + " to "
        + str(round(float(after_scores.mean()), 4))
        + ". A repair ordered on a report that fails more of its standards' requirements was the "
        "more often reported again, so the model must have learned from the screening counts."
    )
    clear = features[features[config.CONFLICTING_STREET_WORKS_COLUMN] == 0]
    beside = clear.copy()
    beside[config.CONFLICTING_STREET_WORKS_COLUMN] = 1
    beside[config.LIVE_STREET_WORKS_COLUMN] = beside[config.LIVE_STREET_WORKS_COLUMN] + 1
    beside[config.LIVE_STREET_WORKS_TRIGGER_COLUMN] = 1
    clear_scores = model.predict_proba(clear)[:, 1]
    beside_scores = model.predict_proba(beside)[:, 1]
    rising = float((beside_scores > clear_scores).mean())
    assert len(clear) > 50 and rising >= 0.95 and (
            beside_scores.mean() - clear_scores.mean()) >= 0.05, (
        "Putting one conflicting street works permit live on the street, with its coordination "
        "trigger flagged, raises the repeat score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(clear))
        + " held-out reports with none, and moves the average from "
        + str(round(float(clear_scores.mean()), 4))
        + " to "
        + str(round(float(beside_scores.mean()), 4))
        + ". A defect repaired where other works dig up the same part of the street is the more "
        "often reported again, so the model must have learned it from the register counts."
    )


def test_expected_hours_rise_on_busy_street_and_conflict(processed_frames, saved_models):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_coordination_trigger_flags, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    logs = processed_frames[config.CLEAN_REPAIR_LOGS_EVAL_FILE]
    local = logs[logs[config.STREET_CATEGORY_COLUMN] == config.LOCAL_ROAD]
    sensitive = local.assign(**{config.STREET_CATEGORY_COLUMN: config.TRAFFIC_SENSITIVE})
    clear = logs[logs[config.CONFLICTING_STREET_WORKS_COLUMN] == 0]
    beside = clear.assign(**{config.CONFLICTING_STREET_WORKS_COLUMN: 1,
                             config.LIVE_STREET_WORKS_COLUMN:
                                 clear[config.LIVE_STREET_WORKS_COLUMN] + 1})
    try:
        flagged = [add_coordination_trigger_flags(frame) for frame in [local, sensitive, clear,
                                                                       beside]]
    except Exception as error:
        flagged = error
    assert isinstance(flagged, list) and all(isinstance(frame, pd.DataFrame)
                                             for frame in flagged), (
        "add_coordination_trigger_flags gave "
        + repr(flagged)[:200]
        + ", where it returns the repair logs with their trigger flags beside them."
    )
    features = [build_feature_frame(frame, config.CREW_HOURS_NUMERIC_FEATURES,
                                    config.CREW_HOURS_CATEGORICAL_FEATURES) for frame in flagged]
    assert all(isinstance(frame, pd.DataFrame) for frame in features), (
        "build_feature_frame returned "
        + type(features[0]).__name__
        + ", where it returns the crew hours model's inputs."
    )
    model = saved_models["crew_hours"]
    for label, shorter, longer, count, minimum_rise in [
            ("on a traffic-sensitive street rather than a local road", features[0], features[1],
             len(local), 0.5),
            ("beside a conflicting street works permit rather than with none live", features[2],
             features[3], len(clear), 0.25)]:
        shorter_hours = model.predict(shorter)
        longer_hours = model.predict(longer)
        lengthened = float((longer_hours > shorter_hours).mean())
        assert count > 50 and lengthened >= 0.9 and (
                longer_hours.mean() - shorter_hours.mean()) >= minimum_rise, (
            "With the repair "
            + label
            + ", the expected crew hours rise for "
            + str(round(100 * lengthened, 1))
            + "% of the "
            + str(count)
            + " held-out repairs and on average from "
            + str(round(float(shorter_hours.mean()), 1))
            + " to "
            + str(round(float(longer_hours.mean()), 1))
            + " crew hours, where they rise for nine repairs in ten and by at least "
            + str(minimum_rise)
            + " hours on average. A repair on a busy street needs traffic management and one "
            "beside other works waits for them, so each takes longer, and the model is fed the "
            "street category, the register counts and the coordination triggers."
        )


def test_bands_and_overrun_follow_thresholds(live_files, saved_models, defect_standards):
    from decimal import Decimal

    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_coordination_trigger_flags,
                                               add_requirement_counts, build_feature_frame)
    from tools.prediction_tools import (describe_expected_crew_hours,
                                        estimate_expected_crew_hours,
                                        estimate_repeat_likelihood)
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import count_live_street_works, screen_report

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    standards = defect_standards["standards"]
    band_ranges = {
        config.HIGH_LIKELIHOOD_BAND: (float(config.HIGH_LIKELIHOOD_MINIMUM), float("inf")),
        config.MEDIUM_LIKELIHOOD_BAND: (float(config.MEDIUM_LIKELIHOOD_MINIMUM),
                                        float(config.HIGH_LIKELIHOOD_MINIMUM)),
        config.LOW_LIKELIHOOD_BAND: (float("-inf"), float(config.MEDIUM_LIKELIHOOD_MINIMUM))}
    bands = {band: 0 for band in band_ranges}
    given = []
    for report_reference in live_files["reports"][config.REPORT_REFERENCE_COLUMN]:
        report = fetch_report(live_files["reports"], report_reference)
        try:
            register_count = count_live_street_works(live_files["register"], report)
            for count_column in [config.LIVE_STREET_WORKS_COLUMN,
                                 config.CONFLICTING_STREET_WORKS_COLUMN]:
                report[count_column] = register_count[count_column]
            screening = screen_report(report, standards)
            likelihood = estimate_repeat_likelihood(
                saved_models["repeat"], saved_models["repeat_schema"], report, screening)
            crew_hours = estimate_expected_crew_hours(
                saved_models["crew_hours"], saved_models["crew_hours_schema"], report, screening)
        except Exception as error:
            screening = likelihood = crew_hours = error
        assert (isinstance(screening, dict)
                and set(config.REQUIREMENT_COUNT_RESULTS) <= set(screening)
                and isinstance(screening.get("coordination_triggers"), list)
                and isinstance(likelihood, dict)
                and {"repeat_likelihood", "likelihood_band"} <= set(likelihood)
                and isinstance(crew_hours, dict)
                and {"expected_crew_hours", "planned_crew_hours", "overrun_expected"}
                <= set(crew_hours)
                and isinstance(crew_hours["expected_crew_hours"], (int, float))), (
            "For "
            + report_reference
            + " the screening, the likelihood and the expected crew hours came back as "
            + repr(screening)[:120]
            + ", "
            + repr(likelihood)[:120]
            + " and "
            + repr(crew_hours)[:120]
            + ", where the screening carries its counts and coordination triggers, the likelihood "
            "its figure and band, and the expected crew hours a number with the planned crew "
            "hours and whether an overrun is expected."
        )
        given.append((report, screening, likelihood, crew_hours))
    try:
        frame = add_coordination_trigger_flags(add_requirement_counts(
            pd.DataFrame([report for report, _screening, _likelihood, _crew_hours in given]),
            standards))
    except Exception as error:
        frame = error
    assert isinstance(frame, pd.DataFrame) and len(frame) == len(given), (
        "add_requirement_counts and add_coordination_trigger_flags gave "
        + repr(frame)[:200]
        + " for the live queue, where they give its reports back in order, with their "
        "requirement counts and trigger flags, prepared as the models were trained."
    )
    repeat_inputs = build_feature_frame(frame, config.REPEAT_NUMERIC_FEATURES,
                                        config.REPEAT_CATEGORICAL_FEATURES)
    crew_hours_inputs = build_feature_frame(frame, config.CREW_HOURS_NUMERIC_FEATURES,
                                            config.CREW_HOURS_CATEGORICAL_FEATURES)
    assert isinstance(repeat_inputs, pd.DataFrame) and isinstance(crew_hours_inputs,
                                                                  pd.DataFrame), (
        "build_feature_frame returned "
        + type(repeat_inputs).__name__
        + ", where it returns a table of the model's inputs."
    )
    scores = saved_models["repeat"].predict_proba(repeat_inputs)[:, 1]
    forecasts = saved_models["crew_hours"].predict(crew_hours_inputs)
    overruns = {True: 0, False: 0}
    for (report, _screening, likelihood, crew_hours), score, forecast in zip(
            given, scores, forecasts, strict=True):
        figure = likelihood["repeat_likelihood"]
        band = likelihood["likelihood_band"]
        lowest, highest = band_ranges.get(band, (None, None))
        assert (isinstance(figure, float) and abs(figure - float(score)) <= 0.005 + 1e-9
                and abs(round(figure * 100) - figure * 100) < 1e-6
                and lowest is not None and lowest <= figure < highest), (
            report[config.REPORT_REFERENCE_COLUMN]
            + " is given a repeat likelihood of "
            + str(figure)
            + ", band "
            + str(band)
            + ", where its saved model scores it "
            + str(round(float(score), 4))
            + ". The figure is the model's score to "
            + str(config.LIKELIHOOD_DECIMALS)
            + " decimals, and its band follows config.HIGH_LIKELIHOOD_MINIMUM ("
            + str(config.HIGH_LIKELIHOOD_MINIMUM)
            + ") and config.MEDIUM_LIKELIHOOD_MINIMUM ("
            + str(config.MEDIUM_LIKELIHOOD_MINIMUM)
            + "), applied to the rounded figure."
        )
        bands[band] += 1
        planned = report[config.PLANNED_CREW_HOURS_COLUMN]
        expected_hours = crew_hours["expected_crew_hours"]
        flag = crew_hours["overrun_expected"]
        flagged_right = (
            (planned is None and flag is None)
            or (planned is not None and flag is (Decimal(str(expected_hours))
                                                 > Decimal(str(planned))
                                                 * config.OVERRUN_RATIO)))
        if planned is not None and isinstance(flag, bool):
            overruns[flag] += 1
        assert (abs(round(expected_hours * 10) - expected_hours * 10) < 1e-6
                and abs(expected_hours - float(forecast)) <= 0.05 + 1e-9
                and crew_hours["planned_crew_hours"] == planned and flagged_right), (
            report[config.REPORT_REFERENCE_COLUMN]
            + " is expected to take "
            + str(expected_hours)
            + " crew hours against "
            + str(planned)
            + " planned, with an overrun expected "
            + str(flag)
            + ", where its saved model gives "
            + str(round(float(forecast), 2))
            + ". The expected crew hours are the model's forecast to "
            + str(config.CREW_HOURS_DECIMALS)
            + " decimal, and overrun_expected is True only when that rounded figure exceeds "
            + "config.OVERRUN_RATIO ("
            + str(config.OVERRUN_RATIO)
            + ") times the planned crew hours."
        )
        expected_text = format(expected_hours, "." + str(config.CREW_HOURS_DECIMALS) + "f")
        planned_text = (None if planned is None
                        else format(planned, "." + str(config.CREW_HOURS_DECIMALS) + "f"))
        wording = {
            None: config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(
                expected_crew_hours=expected_text),
            True: config.OVERRUN_TEMPLATE.format(expected_crew_hours=expected_text,
                                                 planned_crew_hours=planned_text),
            False: config.NO_OVERRUN_TEMPLATE.format(expected_crew_hours=expected_text,
                                                     planned_crew_hours=planned_text)}
        assert describe_expected_crew_hours(crew_hours) == wording[flag], (
            "The expected crew hours of "
            + report[config.REPORT_REFERENCE_COLUMN]
            + " are described as "
            + repr(describe_expected_crew_hours(crew_hours))
            + " where the configuration's words for an overrun expected "
            + str(flag)
            + " give "
            + repr(wording[flag])
            + "."
        )
    assert min(bands.values()) > 0 and min(overruns.values()) > 0, (
        "The live queue's likelihoods fall only in the bands "
        + str(bands)
        + " and its overruns only in "
        + str(overruns)
        + ", so the thresholds cannot all be told apart."
    )
    unplanned = dict(given[0][0])
    unplanned[config.PLANNED_CREW_HOURS_COLUMN] = None
    unplanned_hours = estimate_expected_crew_hours(
        saved_models["crew_hours"], saved_models["crew_hours_schema"], unplanned, given[0][1])
    assert (unplanned_hours["overrun_expected"] is None
            and unplanned_hours["planned_crew_hours"] is None
            and describe_expected_crew_hours(unplanned_hours)
            == config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(
                expected_crew_hours=format(unplanned_hours["expected_crew_hours"],
                                           "." + str(config.CREW_HOURS_DECIMALS) + "f"))), (
        "A report with no planned crew hours is described as "
        + repr(describe_expected_crew_hours(unplanned_hours))
        + " with an overrun expected "
        + str(unplanned_hours["overrun_expected"])
        + ". With nothing planned there is nothing to compare: no overrun is worked out."
    )
    first_report, first_screening = given[0][0], given[0][1]
    other_report = given[1][0]
    for label, estimate_name in [("repeat likelihood", "likelihood"),
                                 ("expected crew hours", "crew_hours")]:
        refusal = None
        try:
            if estimate_name == "likelihood":
                estimate_repeat_likelihood(saved_models["repeat"],
                                           saved_models["repeat_schema"],
                                           other_report, first_screening)
            else:
                estimate_expected_crew_hours(saved_models["crew_hours"],
                                             saved_models["crew_hours_schema"],
                                             other_report, first_screening)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert (refusal is not None
                and first_report[config.REPORT_REFERENCE_COLUMN] in refusal
                and other_report[config.REPORT_REFERENCE_COLUMN] in refusal), (
            "A "
            + label
            + " worked out for "
            + other_report[config.REPORT_REFERENCE_COLUMN]
            + " with the screening of "
            + first_report[config.REPORT_REFERENCE_COLUMN]
            + " gave "
            + str(refusal)
            + ". A screening belongs to one report and is refused for any other, naming both."
        )


def test_sentence_model_macro_f1(processed_frames, saved_models):
    import copy

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from config import config
    from modeling.model_evaluation import evaluate_text_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    train_text, held_out_text, train_answers, held_out_answers = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    model = saved_models["sentence"]
    measured = f1_score(held_out_answers, model.predict(held_out_text), average="macro")
    assert measured >= config.MINIMUM_SENTENCE_MACRO_F1, (
        "The sentence model scored "
        + str(round(measured, 4))
        + " macro F1 on the held-out statement sentences, below the "
        + str(config.MINIMUM_SENTENCE_MACRO_F1)
        + " the configuration sets. Macro F1 weighs the rarest type of sentence as much as the "
        "commonest: a measurement read as background is a reading never held to the report."
    )
    from_call_notes = (sentences.loc[train_text.index, config.DOCUMENT_TYPE_COLUMN]
                       == config.CALL_NOTE)
    probe_text = train_text[from_call_notes].head(3000)
    probe_answers = train_answers[from_call_notes].head(3000)
    evaluation = evaluate_text_classifier(copy.deepcopy(model), probe_text, probe_answers,
                                          held_out_text, held_out_answers)
    missing = sorted({"held_out_macro_f1", "train_macro_f1", "generalisation_gap",
                      "per_class_f1"} - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_text_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the sentence model's scores cannot be read."
    )
    trained_on = f1_score(probe_answers, model.predict(probe_text), average="macro")
    assert (abs(evaluation["held_out_macro_f1"] - measured) < 0.006
            and abs(evaluation["train_macro_f1"] - trained_on) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_macro_f1"] - evaluation["held_out_macro_f1"])) < 0.011), (
        "Given the held-out sentences and the first training sentences of call notes, the "
        "evaluation reports macro F1 "
        + str(evaluation["held_out_macro_f1"])
        + " held out and "
        + str(evaluation["train_macro_f1"])
        + " on training, with a gap of "
        + str(evaluation["generalisation_gap"])
        + ", but the model's own predictions give "
        + str(round(measured, 4))
        + " and "
        + str(round(trained_on, 4))
        + ". evaluate_text_classifier scores the fitted model it is given on the sentences it "
        "is given, without fitting it again."
    )


def test_sentence_predictions_reach_every_type(processed_frames, saved_models):
    from sklearn.model_selection import train_test_split

    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = sorted(set(config.SENTENCE_TYPE_MAPPING.values()))
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    found_types = sorted(str(value) for value in
                         sentences[config.SENTENCE_TARGET_COLUMN].dropna().unique())
    unlabelled = int(sentences[config.SENTENCE_TARGET_COLUMN].isna().sum())
    assert unlabelled == 0 and found_types == vocabulary, (
        "The cleaned sentences hold "
        + str(unlabelled)
        + " with no label, and are labelled "
        + str(found_types)
        + " where the sentence types are "
        + str(vocabulary)
        + "."
    )
    model = saved_models["sentence"]
    learned = sorted(str(label) for label in getattr(model, "classes_", []))
    assert learned == vocabulary, (
        "The sentence model knows the classes "
        + str(learned)
        + " where the sentence types are "
        + str(vocabulary)
        + ". A type of sentence the model cannot predict is a type no Highways Inspector ever "
        "sees."
    )
    split = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    held_out_text, held_out_answers = split[1], split[3]
    predicted = [str(label) for label in model.predict(held_out_text)]
    stray = sorted(set(predicted) - set(vocabulary))
    unreached = sorted(set(vocabulary) - set(predicted))
    assert len(stray) == 0 and len(unreached) == 0, (
        "On the held-out sentences the sentence model predicts "
        + str(stray)
        + " outside the vocabulary and never predicts "
        + str(unreached)
        + "."
    )
    for sentence_type in vocabulary:
        shown = held_out_text[held_out_answers == sentence_type]
        right = sum(1 for label in model.predict(shown) if str(label) == sentence_type)
        assert right >= 0.75 * len(shown), (
            "Of the "
            + str(len(shown))
            + " held-out sentences labelled "
            + sentence_type
            + ", the model reads "
            + str(right)
            + " as that type. Every type of statement sentence is told apart, not only the "
            "common ones."
        )
    unseen = [str(label) for label in model.predict(["Zzq wvx plorth.",
                                                     "No words of this sort were ever written."])]
    assert all(label in vocabulary for label in unseen), (
        "Sentences of words never seen in training were read as "
        + str(unseen)
        + ", outside the sentence types."
    )


def test_statement_text_cleaning_leaves_plain_words(processed_frames, live_files):
    import html

    import pandas as pd

    from config import config
    from data_pipeline.common_cleaning import (convert_markup_to_plain_text,
                                               lower_text_written_in_capitals)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    texts = sentences[config.SENTENCE_TEXT_COLUMN].map(str)
    documents = live_files["documents"][config.DOCUMENT_TEXT_COLUMN].map(str)
    for name, written in [(config.CLEAN_STATEMENT_SENTENCES_FILE, texts),
                          ("the documents attached to the live reports", documents)]:
        left_markup = int(written.str.contains(r"</?[A-Za-z][A-Za-z0-9]*[ />]", regex=True).sum())
        left_entities = int(written.str.contains(r"&[#a-z0-9]+;|\xa0", regex=True).sum())
        doubled_spaces = int(written.str.contains(r"\s{2,}|^\s|\s$", regex=True).sum())
        assert left_markup == 0 and left_entities == 0 and doubled_spaces == 0, (
            name
            + " still hold "
            + str(left_markup)
            + " texts with markup, "
            + str(left_entities)
            + " with escaped characters and "
            + str(doubled_spaces)
            + " with stray spaces. A statement is read as plain words."
        )
    lengths = sentences[config.SENTENCE_LENGTH_COLUMN]
    miscounted = int((lengths != texts.str.len()).sum())
    assert miscounted == 0, (
        str(miscounted)
        + " cleaned sentences carry a length that is not the length of their cleaned text. The "
        "length is counted again once the markup is gone."
    )
    plain = texts[texts.str.contains("[A-Z]", regex=True)
                  & texts.str.contains("[a-z]", regex=True)
                  & ~texts.str.contains("[&<>]", regex=True)].head(40).reset_index(drop=True)
    expected = convert_markup_to_plain_text(plain)
    assert isinstance(expected, pd.Series) and list(expected) == list(plain), (
        "convert_markup_to_plain_text returned "
        + repr(expected)[:200]
        + " for "
        + str(len(plain))
        + " sentences already in plain words, where it returns a series of the same sentences, "
        "one for each, unchanged."
    )
    variants = {
        "wrapped in markup": "<div>" + plain.str.replace(" ", " <br/> ", n=1, regex=False)
                             + "</span>",
        "with its characters escaped": plain.map(html.escape).str.replace(" ", "&nbsp;", n=1,
                                                                          regex=False),
        "with no-break and extra spaces": "  " + plain.str.replace(" ", "\xa0 ", n=1,
                                                                   regex=False).str.replace(
            " ", "   ", regex=False) + " ",
    }
    for label, written in variants.items():
        cleaned = convert_markup_to_plain_text(pd.Series(written))
        assert isinstance(cleaned, pd.Series) and len(cleaned) == len(plain), (
            "convert_markup_to_plain_text returned "
            + type(cleaned).__name__
            + " for sentences "
            + label
            + ", where it returns a series of the same length."
        )
        differing = [(written.iloc[position], cleaned.iloc[position])
                     for position in range(len(plain))
                     if cleaned.iloc[position] != plain.iloc[position]]
        assert len(differing) == 0, (
            "A sentence "
            + label
            + " cleans to something else than the plain sentence, for example "
            + str(differing[:2])
            + ". Markup tags are taken out, character entities read as the characters they stand "
            "for and no-break spaces as spaces, and every run of spaces is one space, leaving the "
            "words the writer meant."
        )
    shouted = lower_text_written_in_capitals(plain.str.upper())
    assert isinstance(shouted, pd.Series) and len(shouted) == len(plain), (
        "lower_text_written_in_capitals returned "
        + type(shouted).__name__
        + " for sentences written in capitals, where it returns a series of the same length."
    )
    differing = [(plain.iloc[position], shouted.iloc[position]) for position in range(len(plain))
                 if shouted.iloc[position] != plain.iloc[position].lower()]
    assert len(differing) == 0, (
        "A sentence written in capitals is lowered to something other than the plain sentence in "
        "small letters, for example "
        + str(differing[:2])
        + ". Capitals carry no meaning the model was trained on, and the case cannot be "
        "restored further than small letters."
    )
    kept_case = lower_text_written_in_capitals(plain)
    assert list(kept_case) == list(plain), (
        "A plain sentence written in mixed case is changed by lowering, for example "
        + str([(plain.iloc[position], kept_case.iloc[position]) for position in range(len(plain))
               if kept_case.iloc[position] != plain.iloc[position]][:2])
        + ". Only a sentence written wholly in capitals is lowered."
    )
    capital_documents = documents.head(5).str.upper().reset_index(drop=True)
    kept_documents = convert_markup_to_plain_text(capital_documents)
    assert list(kept_documents) == list(capital_documents), (
        "A document written in capitals is changed by reading it as plain words, for example "
        + str(list(kept_documents)[:1])[:200]
        + ". A document keeps its own capitals; only a past statement sentence written wholly "
        "in capitals is lowered before the model learns from it."
    )


def test_statements_split_into_ordered_sentences(live_files, saved_models):
    import re

    from config import config
    from tools.evidence_tools import read_document_evidence
    from tools.report_data_tools import fetch_report, fetch_report_documents

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = set(config.SENTENCE_TYPE_MAPPING.values())
    reports = live_files["reports"]
    documents_file = live_files["documents"]
    read_reports = 0
    for report_reference in list(reports[config.REPORT_REFERENCE_COLUMN])[:8]:
        report = fetch_report(reports, report_reference)
        documents = fetch_report_documents(documents_file, report)
        assert isinstance(documents, list) and all(
                isinstance(document, dict) and {config.DOCUMENT_ID_COLUMN,
                                                config.DOCUMENT_TYPE_COLUMN,
                                                config.DOCUMENT_TEXT_COLUMN} <= set(document)
                for document in documents), (
            "fetch_report_documents returned "
            + str(documents)[:200]
            + " for "
            + report_reference
            + ", where it returns the documents attached to the report, each with its "
            "identifier, type and text."
        )
        attached = set(documents_file.loc[documents_file[config.REPORT_REFERENCE_COLUMN]
                                          == report_reference, config.DOCUMENT_ID_COLUMN])
        fetched = [document[config.DOCUMENT_ID_COLUMN] for document in documents]
        assert set(fetched) == attached and len(fetched) == len(attached) and fetched == sorted(
                fetched), (
            "The documents of "
            + report_reference
            + " were fetched as "
            + str(fetched)
            + " where the documents attached to it are "
            + str(sorted(attached))
            + ". Every document attached to a report is fetched once, in the order of the "
            "document identifiers, whatever order they stand in the file."
        )
        try:
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            findings = error
        assert isinstance(findings, list) and all(
                isinstance(finding, dict) and {"document_id", "document_type", "sentence",
                                               "sentence_type"} <= set(finding)
                and isinstance(finding["sentence"], str) for finding in findings), (
            "read_document_evidence gave "
            + repr(findings)[:200]
            + " for the documents of "
            + report_reference
            + ", where it returns one finding for each sentence, with its document's identifier "
            "and type, the sentence and the type the model reads in it."
        )
        read_reports += 1
        runs = [finding["document_id"] for position, finding in enumerate(findings)
                if position == 0 or findings[position - 1]["document_id"] != finding["document_id"]]
        assert runs == fetched, (
            "The sentences of "
            + report_reference
            + " come from the documents in the runs "
            + str(runs)
            + " where the documents run "
            + str(fetched)
            + ". Each document's sentences are read together, in the order written, and the "
            "documents in their own order."
        )
        for document in documents:
            own = [finding["sentence"] for finding in findings
                   if finding["document_id"] == document[config.DOCUMENT_ID_COLUMN]]
            assert " ".join(own) == document[config.DOCUMENT_TEXT_COLUMN], (
                "The sentences read from "
                + document[config.DOCUMENT_ID_COLUMN]
                + " join back to "
                + repr(" ".join(own)[:200])
                + " instead of the document's own text. A sentence is kept whole, as written."
            )
            unsplit = [sentence for sentence in own if re.search(r"[.!?]\s+\S", sentence)]
            cut_short = ([sentence for sentence in own[:-1]
                          if not sentence.endswith((".", "!", "?"))]
                         + [sentence for sentence in own[1:] if not re.match("[A-Za-z0-9]",
                                                                             sentence)])
            assert len(unsplit) == 0 and len(cut_short) == 0, (
                "The document "
                + document[config.DOCUMENT_ID_COLUMN]
                + " was split into pieces that still hold the end of a sentence, "
                + str(unsplit[:2])
                + ", or that were cut where no sentence ends, "
                + str(cut_short[:2])
                + ". A sentence ends at a full stop, question mark or exclamation mark followed "
                "by a space and the next word, and only there."
            )
        for finding in findings:
            assert (finding.get("sentence_type") in vocabulary
                    and finding.get("document_type") == [
                        document[config.DOCUMENT_TYPE_COLUMN] for document in documents
                        if document[config.DOCUMENT_ID_COLUMN] == finding["document_id"]][0]), (
                "A sentence of "
                + str(finding.get("document_id"))
                + " is read as "
                + repr(finding.get("sentence_type"))
                + " in a document of the type "
                + repr(finding.get("document_type"))
                + ". Each sentence carries the type the model reads in it and the type of the "
                "document it comes from."
            )
    assert read_reports == 8, "Fewer than eight live reports could be read."
    assert read_document_evidence(saved_models["sentence"], []) == [], (
        "A report with no document attached gives statement evidence. With nothing attached, "
        "nothing is read."
    )
    lowered_text = ("the hole is 45 mm deep at its deepest point. does the water stand across "
                    "the road after rain? the slab rocks when you step on it!")
    lowered = read_document_evidence(saved_models["sentence"], [
        {config.DOCUMENT_ID_COLUMN: "DOC-990001",
         config.DOCUMENT_TYPE_COLUMN: config.RESIDENT_DESCRIPTION,
         config.DOCUMENT_TEXT_COLUMN: lowered_text}])
    pieces = ([finding.get("sentence") for finding in lowered if isinstance(finding, dict)]
              if isinstance(lowered, list) else [])
    assert len(pieces) == 3 and " ".join(str(piece) for piece in pieces) == lowered_text, (
        "A document of three sentences written in small letters, as a resident may write one, "
        "was split into "
        + str(pieces)
        + ". A sentence ends at its full stop, question mark or exclamation mark whatever the "
        "case of the next word."
    )
    refusal = None
    try:
        read_document_evidence(saved_models["sentence"], [
            {config.DOCUMENT_ID_COLUMN: "DOC-990002",
             config.DOCUMENT_TYPE_COLUMN: config.CALL_NOTE,
             config.DOCUMENT_TEXT_COLUMN: None}])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "DOC-990002" in refusal, (
        "A document whose text is blank gave "
        + str(refusal)
        + ". A document with nothing in it cannot be read and is refused with a ValueError "
        "naming the document."
    )


def test_sentences_link_requirements_and_readings(live_files, saved_models, defect_standards):
    from config import config
    from tools.evidence_tools import (find_disagreements, link_evidence_to_requirements,
                                      read_document_evidence)
    from tools.report_data_tools import fetch_report, fetch_report_documents
    from tools.screening_tools import count_live_street_works, screen_report

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    standards = defect_standards["standards"]
    reports = live_files["reports"]
    seen_types = set()
    for report_reference in reports[config.REPORT_REFERENCE_COLUMN]:
        report = fetch_report(reports, report_reference)
        documents = fetch_report_documents(live_files["documents"], report)
        if report[config.DEFECT_TYPE_COLUMN] in seen_types or not documents:
            continue
        seen_types.add(report[config.DEFECT_TYPE_COLUMN])
        try:
            register_count = count_live_street_works(live_files["register"], report)
            for count_column in [config.LIVE_STREET_WORKS_COLUMN,
                                 config.CONFLICTING_STREET_WORKS_COLUMN]:
                report[count_column] = register_count[count_column]
            screening = screen_report(report, standards)
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            screening = findings = error
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and isinstance(findings, list)
                and all(isinstance(finding, dict) for finding in findings)), (
            "For "
            + report_reference
            + " the screening came back as "
            + repr(screening)[:150]
            + " and the statement evidence as "
            + repr(findings)[:150]
            + ", where the screening holds its standards' requirements and the evidence one "
            "finding for each sentence."
        )
        linked = link_evidence_to_requirements(findings, screening)
        assert isinstance(linked, list) and len(linked) == len(findings), (
            "link_evidence_to_requirements returned "
            + str(linked)[:150]
            + " for the "
            + str(len(findings))
            + " sentences of "
            + report_reference
            + ", where it returns one finding for each."
        )
        own_requirements = [{"defect_type": requirement.get("defect_type"),
                             "requirement_name": requirement.get("requirement_name")}
                            for requirement in screening["requirements"]]
        for finding, linked_finding in zip(findings, linked, strict=True):
            borne = linked_finding.get("requirements_borne_on")
            assert (isinstance(borne, list)
                    and borne == [requirement for requirement in own_requirements
                                  if requirement in borne]
                    and len(borne) == len({str(requirement) for requirement in borne})
                    and (finding.get("sentence_type") != config.BACKGROUND or borne == [])
                    and linked_finding.get("sentence") == finding.get("sentence")
                    and linked_finding.get("sentence_type") == finding.get("sentence_type")
                    and linked_finding.get("document_id") == finding.get("document_id")), (
                "A sentence of "
                + report_reference
                + " read as "
                + str(finding.get("sentence_type"))
                + " bears on "
                + str(borne)
                + " where the report's own standards have the requirements "
                + str(own_requirements)
                + ". A sentence bears only on requirements its report's own standards have, "
                "each named with its defect type, in the screening's order, background on none, "
                "and keeps its sentence, type and document."
            )
    assert seen_types == set(config.DEFECT_STANDARD_DOCUMENTS), (
        "The live reports read covered the defect types "
        + str(sorted(seen_types))
        + ", where every defect type of the configuration has a live report with a document."
    )
    cw, fw, dr = config.CARRIAGEWAY, config.FOOTWAY, config.DRAINAGE
    vg, sl = config.VEGETATION, config.STREET_LIGHTING
    named = {"report_reference": "SC0950110", "defect_types": [cw, fw, dr, vg, sl],
             "requirements": []}
    for defect_type, kind, name in [
            (cw, config.LOCATION_FIXED_REQUIREMENT, "Pinned spot"),
            (cw, config.PHOTOGRAPH_REQUIREMENT, "Snapshot"),
            (cw, config.DEFECT_DEPTH_REQUIREMENT, "Hole depth"),
            (cw, config.DEFECT_WIDTH_REQUIREMENT, "Hole size"),
            (fw, config.DEFECT_DEPTH_REQUIREMENT, "Trip edge"),
            (fw, config.DEFECT_WIDTH_REQUIREMENT, "Slab gap"),
            (dr, config.WATER_DEPTH_REQUIREMENT, "Puddle depth"),
            (dr, config.WATER_REACH_REQUIREMENT, "Puddle spread"),
            (vg, config.HIGHWAY_LAND_REQUIREMENT, "Verge land"),
            (vg, config.CLEAR_HEIGHT_REQUIREMENT, "Branch headroom"),
            (sl, config.COLUMN_NUMBER_REQUIREMENT, "Post plate"),
            (sl, config.COLUMN_LEAN_REQUIREMENT, "Post tilt")]:
        named["requirements"].append({"defect_type": defect_type, "requirement_kind": kind,
                                      "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "result": config.REQUIREMENT_MET, "blank_answers": []})
    wanted = {
        config.MEASUREMENT: [(cw, "Hole depth"), (cw, "Hole size"), (fw, "Trip edge"),
                             (fw, "Slab gap"), (dr, "Puddle depth"), (dr, "Puddle spread"),
                             (vg, "Branch headroom"), (sl, "Post tilt")],
        config.LOCATION: [(cw, "Pinned spot"), (vg, "Verge land"), (sl, "Post plate")],
        config.PHOTOGRAPH: [(cw, "Snapshot")],
        config.WEATHER_AND_WATER: [(dr, "Puddle depth"), (dr, "Puddle spread")],
        config.HAZARD: [], config.INCIDENT: [], config.TRAFFIC_AND_USE: [],
        config.PREVIOUS_REPORTS: [], config.BACKGROUND: []}
    wanted = {sentence_type: [{"defect_type": defect_type, "requirement_name": name}
                              for defect_type, name in pairs]
              for sentence_type, pairs in wanted.items()}
    linked = link_evidence_to_requirements(
        [{"document_id": "DOC-990003", "document_type": config.RESIDENT_DESCRIPTION,
          "sentence": "A sentence read as " + sentence_type + ".", "sentence_type": sentence_type}
         for sentence_type in wanted], named)
    got = ({finding.get("sentence_type"): finding.get("requirements_borne_on")
            for finding in linked if isinstance(finding, dict)}
           if isinstance(linked, list) else linked)
    assert got == wanted, (
        "Under standards whose requirements carry names no Wrenford standard uses, one sentence "
        "of each type bears on "
        + str(got)
        + " where they bear on "
        + str(wanted)
        + ". Each type of sentence bears on the requirements of the kinds its statement guide "
        "names, found by their kind, in the screening's own order and named with their defect "
        "type, and hazard, incident, traffic and use, previous reports and background on none."
    )
    refusal = None
    try:
        link_evidence_to_requirements(
            [{"document_id": "DOC-990004", "document_type": config.CALL_NOTE,
              "sentence": "A sentence of a type nobody configured.", "sentence_type": "Rumour"}],
            named)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "Rumour" in refusal, (
        "A sentence of a type the configuration gives no requirements for gave "
        + str(refusal)
        + ". It is refused with a ValueError naming the type."
    )
    measurement = config.MEASUREMENT
    depth, width, water = "defect_depth_mm", "defect_width_mm", "water_depth_mm"
    clear = config.CLEAR_HEIGHT_COLUMN
    disagrees, blank = config.DISAGREES_WITH_FORM, config.DESCRIBES_BLANK_ANSWER
    footway_sentences = [
        (measurement, "The hole is 45 mm deep at its deepest point."),
        (measurement, "It is 38 mm deep at the kerb."),
        (measurement, "The gap measures 300 mm wide."),
        (measurement, "There is 60 mm of standing water in it."),
        (measurement, "The slab is raised but nobody measured it."),
        (config.BACKGROUND, "Last year it was 70 mm deep before the repair."),
        (measurement, "Measured again: 45.4 MM DEEP and 300 mm wide."),
        (config.HAZARD, "A cyclist swerved round the 50 mm deep edge.")]
    vegetation_sentences = [
        (measurement, "There is only 2.1 m clear under the branches."),
        (measurement, "Two readings, 2.4 m clear and 2.40 m clear, were taken."),
        (measurement, "The branch hangs 2.4 m clear and the root lifts the slab 20 mm deep.")]
    footway_report = {config.REPORT_REFERENCE_COLUMN: "SC0950111", depth: 45, width: 300,
                      water: None, clear: None}
    vegetation_report = {config.REPORT_REFERENCE_COLUMN: "SC0950112", depth: None, width: None,
                         water: None, clear: 2.4}
    cases = [
        ("a footway report 45 mm deep and 300 mm wide, with no standing water given",
         footway_sentences, footway_report,
         [(footway_sentences[1][1], {depth: [38.0]}, {depth: 45.0}, disagrees),
          (footway_sentences[3][1], {water: [60.0]}, {water: None}, blank),
          (footway_sentences[6][1], {depth: [45.4], width: [300.0]},
           {depth: 45.0, width: 300.0}, disagrees)]),
        ("a vegetation report 2.4 m clear, with no depth given", vegetation_sentences,
         vegetation_report,
         [(vegetation_sentences[0][1], {clear: [2.1]}, {clear: 2.4}, disagrees),
          (vegetation_sentences[2][1], {clear: [2.4], depth: [20.0]}, {clear: 2.4, depth: None},
           blank)])]
    for meaning, sentences, report, expected in cases:
        findings = [{"document_id": "DOC-990005", "document_type": config.RESIDENT_DESCRIPTION,
                     "sentence": sentence, "sentence_type": sentence_type}
                    for sentence_type, sentence in sentences]
        flagged = find_disagreements(findings, report)
        assert isinstance(flagged, list) and all(isinstance(flag, dict) for flag in flagged), (
            "find_disagreements returned "
            + str(flagged)[:200]
            + ", where it returns one flag for each measurement sentence it flags."
        )
        got = [(flag.get("sentence"), flag.get("stated_readings"), flag.get("report_readings"),
                flag.get("flag")) for flag in flagged]
        right_document = all(flag.get("document_id") == "DOC-990005" for flag in flagged)
        assert got == expected and right_document, (
            "For "
            + meaning
            + ", the measurement sentences were flagged as "
            + str(got)
            + " where the Report Rules flag "
            + str(expected)
            + ". A figure written as a number and mm deep, mm wide, mm of standing water or m "
            "clear, in any capitals, is held as written, without rounding, to the report's reading "
            "of the same answer: a sentence with no figure is not flagged, one with a figure for "
            "an answer the report leaves blank describes a blank answer, any other difference "
            "disagrees with the form, and only measurement sentences are read."
        )


def test_every_document_gives_up_text(document_texts, tmp_path):
    import os
    import shutil

    import pypdf

    from config import config
    from knowledge.pdf_extraction import extract_text_from_directory

    assert document_texts["problem"] is None, document_texts["problem"]
    in_folder = sorted(name for name in os.listdir(config.DOCS_DIRECTORY)
                       if name.lower().endswith(".pdf"))
    documents = document_texts["documents"]
    assert len(documents) > 0 and {"document_name", "document_text"} <= set(documents[0]), (
        "Reading the document folder returned "
        + str(len(documents))
        + " entries"
        + ("" if len(documents) == 0 else ", the first keyed " + str(sorted(documents[0])))
        + ", where every entry names its document and carries its text."
    )
    names = sorted(document["document_name"] for document in documents)
    assert names == in_folder, (
        "Reading the document folder returned "
        + str(names)
        + " where the folder holds "
        + str(in_folder)
        + ". A document never read is a question answered from general knowledge instead."
    )
    for document in documents:
        assert len(document["document_text"].split()) > 250, (
            document["document_name"]
            + " gave up only "
            + str(len(document["document_text"].split()))
            + " words. A PDF that will not give up its text is indexed as an empty passage."
        )
    for required in (list(config.DEFECT_STANDARD_DOCUMENTS.values())
                     + list(config.SENTENCE_TYPE_GUIDE_DOCUMENTS.values())
                     + config.CHAT_EXCLUDED_DOCUMENTS):
        assert required in names, (
            required
            + " was not read, and the configuration reads it by name."
        )
    folder = tmp_path / "mixed_folder"
    folder.mkdir()
    standard_document = config.DEFECT_STANDARD_DOCUMENTS[config.STREET_LIGHTING]
    guide_document = config.SENTENCE_TYPE_GUIDE_DOCUMENTS[config.MEASUREMENT]
    for name in [standard_document, guide_document]:
        shutil.copy(os.path.join(config.DOCS_DIRECTORY, name), str(folder / name))
    (folder / "reading-list.txt").write_text("Not a document of the street care office.")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    mixed_names = None
    try:
        mixed_names = sorted(entry["document_name"]
                             for entry in extract_text_from_directory(str(folder)))
    except Exception as error:
        mixed_names = type(error).__name__ + ": " + str(error)[:200]
    assert mixed_names == sorted([standard_document, guide_document]), (
        "A folder holding two PDFs, a text file and an image was read as "
        + str(mixed_names)
        + ". Only the PDFs are the documents; anything else in the folder is passed over."
    )
    blank_folder = tmp_path / "blank_page_folder"
    blank_folder.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PdfReader(os.path.join(config.DOCS_DIRECTORY,
                                                 standard_document)).pages[0])
    writer.add_blank_page()
    with open(str(blank_folder / "Standard-With-A-Blank-Page.pdf"), "wb") as blank_file:
        writer.write(blank_file)
    empty_folder = tmp_path / "empty_folder"
    empty_folder.mkdir()
    (empty_folder / "notes.txt").write_text("No PDF here.")
    for label, folder_path, named in [
            ("A PDF whose second page gives up no text", blank_folder,
             "Standard-With-A-Blank-Page.pdf"),
            ("A folder with no PDF in it", empty_folder, str(empty_folder))]:
        refusal = None
        try:
            extract_text_from_directory(str(folder_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and named in refusal, (
            label
            + " gave "
            + str(refusal)
            + ". A page that cannot be read may be the one that states a requirement, and a "
            "folder with no document has nothing to answer from, so each is refused with a "
            "ValueError naming the file or the folder."
        )


def test_documents_cut_into_sections_and_passages(document_passages, defect_standards):
    import re

    from config import config
    from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

    assert document_passages["problem"] is None, document_passages["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    documents = document_passages["documents"]
    sections = document_passages["sections"]
    chunks = document_passages["chunks"]
    assert len(chunks) > 0 and {"chunk_id", "chunk_text", "document_name"} <= set(chunks[0]), (
        "Cutting the documents returned "
        + str(len(chunks))
        + " passages"
        + ("" if len(chunks) == 0 else ", the first keyed " + str(sorted(chunks[0])))
        + ", where every passage carries its identifier, its text and its document."
    )
    assert len(sections) > 5 * len(documents), (
        str(len(documents))
        + " documents split into "
        + str(len(sections))
        + " sections. Each document is written in numbered sections, and a section is the unit "
        "the tools read and a question is usually about."
    )
    with_header = [chunk["chunk_id"] for chunk in chunks
                   if re.search(config.DOCUMENT_HEADER_PATTERN, chunk["chunk_text"])]
    assert len(with_header) == 0, (
        "These passages still carry the document header: "
        + str(with_header[:5])
        + ". The header repeats the same words in every document, so it matches every question "
        "equally and pushes the passages that answer it out of the results."
    )
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    average = sum(lengths) / len(lengths)
    assert (config.CHUNK_SIZE_CHARACTERS / 5 <= average <= config.CHUNK_SIZE_CHARACTERS
            and max(lengths) <= config.CHUNK_SIZE_CHARACTERS), (
        "Passages average "
        + str(int(average))
        + " characters, the longest "
        + str(max(lengths))
        + ", where the configuration asks for passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + ". Passages far longer return a whole document for every question; far shorter lose "
        "the sentence that gave them their meaning."
    )
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    assert len(identifiers) == len(set(identifiers)), (
        "Passage identifiers repeat, so storing them would overwrite one passage with another."
    )
    words_by_document = {document["document_name"]: set(document["document_text"].split())
                         for document in documents}
    for chunk in chunks:
        words = chunk["chunk_text"].split()
        foreign = [word for word in words
                   if word not in words_by_document.get(chunk["document_name"], set())]
        assert len(words) > 0 and len(foreign) <= len(words) // 10, (
            "Passage "
            + str(chunk["chunk_id"])
            + " says it comes from "
            + str(chunk["document_name"])
            + " but "
            + str(len(foreign))
            + " of its "
            + str(len(words))
            + " words are not in that document, for example "
            + str(foreign[:5])
            + ". A passage has to cite the document it was cut from."
        )
    for defect_type, standard in defect_standards["standards"].items():
        own_chunks = [re.sub(r"\s+", " ", chunk["chunk_text"]) for chunk in chunks
                      if chunk["document_name"] == config.DEFECT_STANDARD_DOCUMENTS[defect_type]]
        broken = [requirement["requirement_sentence"] for requirement in standard["requirements"]
                  if not any(requirement["requirement_sentence"] in chunk_text
                             for chunk_text in own_chunks)]
        assert len(broken) == 0, (
            "These requirements of "
            + config.DEFECT_STANDARD_DOCUMENTS[defect_type]
            + " are not whole inside any one of its passages: "
            + str(broken[:3])
            + ". A requirement cut across two passages can no longer be read from either."
        )
    header = ("Document Standard 99 | Issue 1 | Owner: Highways Asset Management | Review cycle: "
              "every six months")
    lines = []
    for first in "abcdefgh":
        for second in "abcdefg":
            lines.append("Reading " + first + second + " of the overlap check names break "
                         + first + second + ".")
    synthetic = [{"document_name": "Overlap-Check.pdf",
                  "document_text": "WRENFORD BOROUGH COUNCIL\nOverlap Check\n" + header
                                   + "\n1. Overlap check\n" + " ".join(lines)}]
    synthetic_chunks = build_chunk_records(split_documents_into_sections(synthetic))
    assert isinstance(synthetic_chunks, list) and all(
            isinstance(chunk, dict) and "chunk_text" in chunk for chunk in synthetic_chunks), (
        "Cutting a document of one long section returned "
        + str(synthetic_chunks)[:200]
        + ", where it returns a list of passages, each with its text."
    )
    pieces = [str(chunk["chunk_text"]) for chunk in synthetic_chunks]
    assert len(pieces) >= 3, (
        "One numbered section of "
        + str(len(" ".join(lines)))
        + " characters was cut into "
        + str(len(pieces))
        + " passages, where passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " characters need at least three."
    )
    unshared = [position for position in range(1, len(pieces))
                if pieces[position].strip()[:20] not in pieces[position - 1]]
    assert len(unshared) == 0, (
        "Passages "
        + str(unshared)
        + " of one long section open with text the passage before them does not carry. "
        "Neighbouring passages of a section overlap by up to "
        + str(config.CHUNK_OVERLAP_CHARACTERS)
        + " characters, so a sentence cut at a boundary is still whole in one of them."
    )
    for description, document_text in [
            ("no header line", "WRENFORD BOROUGH COUNCIL\n1. Overlap check\n" + " ".join(lines)),
            ("two header lines", header + "\n" + header + "\n1. Overlap check\n"
             + " ".join(lines)),
            ("sections numbered from 2", header + "\n2. Overlap check\n" + " ".join(lines))]:
        refusal = None
        try:
            split_documents_into_sections([{"document_name": "Broken-Layout.pdf",
                                            "document_text": document_text}])
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and "Broken-Layout.pdf" in refusal, (
            "A document with "
            + description
            + " was split, giving "
            + str(refusal)
            + ". Every document carries one header line and sections numbered from 1, so a "
            "document laid out otherwise has been read wrongly and is refused, naming it."
        )


def test_index_holds_every_passage_once(knowledge_collection, document_passages):
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    documents = document_passages["documents"]
    chunks = document_passages["chunks"]
    stored = collection.get(include=["metadatas", "documents"])
    assert collection.count() == len(chunks) == len(stored["ids"]), (
        "The knowledge index holds "
        + str(collection.count())
        + " passages where cutting the documents gives "
        + str(len(chunks))
        + ". More means the index was added to twice and repeats itself in every result; fewer "
        "means passages no search can reach."
    )
    assert len(set(stored["ids"])) == len(stored["ids"]), (
        "Stored passage identifiers repeat, so one passage has overwritten another and its text "
        "can no longer be found."
    )
    names = {document["document_name"] for document in documents}
    cited = [(metadata or {}).get("document_name") for metadata in stored["metadatas"]]
    unnamed = [value for value in cited if value not in names]
    assert len(unnamed) == 0, (
        str(len(unnamed))
        + " stored passages name no document from the folder, for example "
        + str(unnamed[:3])
        + ". The name is what lets an answer cite its source and a section be read from its "
        "own document alone."
    )
    by_document = {}
    for chunk in chunks:
        by_document[chunk["document_name"]] = by_document.get(chunk["document_name"], 0) + 1
    stored_by_document = {name: cited.count(name) for name in names}
    assert stored_by_document == by_document, (
        "Passages stored per document "
        + str(stored_by_document)
        + " differ from passages cut per document "
        + str(by_document)
        + ". A document with passages missing answers less than it says."
    )
    assert sorted(stored["documents"]) == sorted(chunk["chunk_text"] for chunk in chunks), (
        "The texts in the knowledge index are not the passages the documents are cut into, so "
        "a section read from the index is not the section the document states."
    )
    for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"],
                                        strict=True):
        twins = [chunk for chunk in chunks if chunk["chunk_id"] == chunk_id]
        assert (len(twins) == 1 and twins[0]["chunk_text"] == text
                and twins[0]["document_name"] == (metadata or {}).get("document_name")), (
            "The stored passage "
            + str(chunk_id)
            + " does not carry the text and document of the passage cut under that identifier."
        )


def test_tools_read_whole_own_sections(knowledge_collection, document_passages):
    from config import config
    from knowledge.vector_index import read_document_section
    from tools.retrieval_tools import (read_inspection_records, read_policy_section,
                                       read_resident_update_guidance, read_reviewer_checks,
                                       read_works_order_guidance)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    unread = []
    for section in document_passages["sections"]:
        read_back = read_document_section(collection, section["document_name"],
                                          section["section_number"])
        if read_back != section["section_text"]:
            unread.append(section["document_name"] + " section "
                          + str(section["section_number"]) + " read as "
                          + repr(str(read_back)[:120]))
    assert len(unread) == 0, (
        str(len(unread))
        + " sections of the "
        + str(len(document_passages["sections"]))
        + " read back from the index differ from the documents, for example "
        + str(unread[:2])
        + ". A section is read back from its own document's passages alone, whole and only "
        "once, exactly as the documents were divided."
    )
    wanted = []
    for defect_type, document_name in config.DEFECT_STANDARD_DOCUMENTS.items():
        wanted.append(("what an inspection of a " + defect_type + " defect records",
                       document_name, config.INSPECTION_RECORDS_SECTION_NUMBER,
                       config.INSPECTION_RECORDS_SECTION_TITLE,
                       read_inspection_records(collection, defect_type)))
        wanted.append(("what the works order of a " + defect_type + " defect says", document_name,
                       config.WORKS_ORDER_SECTION_NUMBER, config.WORKS_ORDER_SECTION_TITLE,
                       read_works_order_guidance(collection, defect_type)))
        wanted.append(("what the resident update of a " + defect_type + " defect says",
                       document_name, config.RESIDENT_UPDATE_SECTION_NUMBER,
                       config.RESIDENT_UPDATE_SECTION_TITLE,
                       read_resident_update_guidance(collection, defect_type)))
    for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items():
        wanted.append(("what the reviewer checks in a sentence of the type " + sentence_type,
                       document_name, config.REVIEWER_CHECKS_SECTION_NUMBER,
                       config.REVIEWER_CHECKS_SECTION_TITLE,
                       read_reviewer_checks(collection, sentence_type)))
    for section_number, section_title in [
            (config.POLICY_TRIGGERS_SECTION_NUMBER, config.POLICY_TRIGGERS_SECTION_TITLE),
            (config.POLICY_DECISIONS_SECTION_NUMBER, config.POLICY_DECISIONS_SECTION_TITLE),
            (config.POLICY_DOCUMENTS_SECTION_NUMBER, config.POLICY_DOCUMENTS_SECTION_TITLE)]:
        wanted.append(("the Highways Policy's section " + section_title,
                       config.HIGHWAYS_POLICY_DOCUMENT, section_number, section_title,
                       read_policy_section(collection, section_number, section_title)))
    for label, document_name, section_number, section_title, passage in wanted:
        heading = str(section_number) + ". " + section_title + " "
        sections = [section["section_text"] for section in document_passages["sections"]
                    if section["document_name"] == document_name
                    and section["section_text"].startswith(heading)]
        assert len(sections) == 1, (
            document_name
            + " holds "
            + str(len(sections))
            + " sections headed "
            + repr(heading.strip())
            + " where it holds one."
        )
        expected = sections[0][len(heading):]
        assert passage == expected, (
            "Reading "
            + label
            + " gave "
            + repr(str(passage)[:160])
            + " where "
            + document_name
            + "'s section "
            + repr(heading.strip())
            + " reads "
            + repr(expected[:160])
            + ". The tools read that section whole, from that document alone, without its "
            "heading."
        )
    for label, call_name, unknown in [
            ("what an inspection records for a defect type with no standard", "inspection",
             "Graffiti"),
            ("the works order of a defect type with no standard", "order", "Fly tipping"),
            ("the resident update of a defect type with no standard", "update", "Road markings"),
            ("the reviewer's checks for a sentence type with no guide", "checks", "Rumour"),
            ("a policy section under a title it does not carry", "policy",
             "The street parties"),
            ("a section the index does not hold", "section",
             config.DEFECT_STANDARD_DOCUMENTS[config.CARRIAGEWAY])]:
        refusal = None
        try:
            if call_name == "inspection":
                read_inspection_records(collection, unknown)
            elif call_name == "order":
                read_works_order_guidance(collection, unknown)
            elif call_name == "update":
                read_resident_update_guidance(collection, unknown)
            elif call_name == "checks":
                read_reviewer_checks(collection, unknown)
            elif call_name == "policy":
                read_policy_section(collection, config.POLICY_TRIGGERS_SECTION_NUMBER, unknown)
            else:
                read_document_section(collection, unknown, 40)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and unknown in refusal, (
            "Asking for "
            + label
            + ", "
            + unknown
            + ", gave "
            + str(refusal)
            + ". It is refused with a ValueError naming it, never answered from another "
            "document or section."
        )


def test_chat_search_skips_casebook_nearest_first(knowledge_collection):
    from config import config
    from knowledge.vector_index import search_documents

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    stored = collection.get(include=["documents", "metadatas"])
    casebook_passages = [passage_id for passage_id, metadata
                         in zip(stored["ids"], stored["metadatas"], strict=True)
                         if (metadata or {}).get("document_name") in config.CHAT_EXCLUDED_DOCUMENTS]
    assert len(casebook_passages) > 0, (
        "The index holds no passage of "
        + str(config.CHAT_EXCLUDED_DOCUMENTS)
        + ", which is indexed with the other documents and left out of the chat's search only."
    )
    longest = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"], strict=True):
        document_name = (metadata or {}).get("document_name")
        if (document_name not in config.CHAT_EXCLUDED_DOCUMENTS
                and len(text) > len(longest.get(document_name, ""))):
            longest[document_name] = text
    questions = [(text, True) for text in longest.values()]
    questions.append(("What did the inspector decide when a past report was referred for "
                      "coordination?", False))
    questions.append(("SC0041207 was phoned in for a pothole outside the post office on Mill "
                      "Lane.", False))
    results = search_documents(collection, [pair[0] for pair in questions])
    assert isinstance(results, list) and len(results) == len(questions), (
        "search_documents answered "
        + str(results)[:200]
        + " for "
        + str(len(questions))
        + " questions, where it answers one list of passages a question, in order."
    )
    for (question, stored_passage), passages in zip(questions, results, strict=True):
        assert (isinstance(passages, list) and len(passages) == config.CHAT_RETRIEVAL_RESULT_COUNT
                and all(isinstance(passage, dict)
                        and {"chunk_id", "chunk_text", "document_name", "distance"} <= set(passage)
                        for passage in passages)), (
            "The search for "
            + repr(question[:80])
            + " returned "
            + str(passages)[:200]
            + " where it returns "
            + str(config.CHAT_RETRIEVAL_RESULT_COUNT)
            + " passages, each with its identifier, text, document and distance."
        )
        distances = [float(passage["distance"]) for passage in passages]
        found = [passage["document_name"] for passage in passages]
        assert distances == sorted(distances) and all(-0.0001 <= value <= 2.0001
                                                      for value in distances), (
            "The search for "
            + repr(question[:80])
            + " returned distances "
            + str([round(value, 3) for value in distances])
            + ". Passages come nearest first, at cosine distances between 0 and 2."
        )
        excluded = [name for name in found if name in config.CHAT_EXCLUDED_DOCUMENTS]
        assert len(excluded) == 0, (
            "The search for "
            + repr(question[:80])
            + " returned passages of "
            + str(excluded)
            + ". The casebook records past reviews of named reports and is never quoted by the "
            "chat, however close its passages are."
        )
        if stored_passage:
            assert passages[0]["chunk_text"] == question, (
                "Searching with the words of the stored passage "
                + repr(question[:80])
                + " returned "
                + repr(str(passages[0]["chunk_text"])[:80])
                + " first. No passage is nearer to a passage's own words than the passage "
                "itself, so it comes back first."
            )


def test_each_defect_type_read_from_own_standard(invented_standards):
    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standards = invented_standards["standards"]
    assert sorted(standards) == sorted(config.DEFECT_STANDARD_DOCUMENTS), (
        "Standards were read for "
        + str(sorted(standards))
        + " where every one of the five defect types has its own."
    )
    for defect_type, requirements in invented_standards["invented"].items():
        standard = standards[defect_type]
        document_name = config.DEFECT_STANDARD_DOCUMENTS[defect_type]
        assert (standard.get("defect_type") == defect_type
                and standard.get("document_name") == document_name), (
            "The standard read for "
            + defect_type
            + " names the defect type "
            + repr(standard.get("defect_type"))
            + " and the document "
            + repr(standard.get("document_name"))
            + ". A standard is read from the document configured for its own defect type and "
            "names both."
        )
        read = [(requirement.get("requirement_name"), requirement.get("requirement_kind"))
                for requirement in standard.get("requirements", [])]
        stated = [(requirement[1], requirement[0]) for requirement in requirements]
        assert read == stated, (
            "A standard written for the checks under the name "
            + document_name
            + " was read as the requirements "
            + str(read)
            + " where its section 4 states "
            + str(stated)
            + ", in that order. Each numbered requirement is read whole, named as the standard "
            "names it, from the standard of its own defect type alone: the carriageway and "
            "footway standards read alike, and neither lends the other a requirement."
        )


def test_requirements_keep_sentence_and_figures(invented_standards):
    from decimal import Decimal

    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    lowest, highest = config.MINIMUM_RULE, config.MAXIMUM_RULE
    over_footway, over_carriageway = config.FOOTWAY_LIMIT_RULE, config.CARRIAGEWAY_LIMIT_RULE
    stated = {
        config.CARRIAGEWAY: [{}, {}, {lowest: Decimal("35")},
                             {lowest: Decimal("120"), highest: Decimal("1300")}],
        config.FOOTWAY: [{}, {lowest: Decimal("15")},
                         {lowest: Decimal("110"), highest: Decimal("1100")}, {}],
        config.DRAINAGE: [{}, {}, {lowest: Decimal("45")}, {lowest: Decimal("1.3")}],
        config.VEGETATION: [{}, {}, {over_footway: Decimal("2.2"),
                                     over_carriageway: Decimal("4.9")}],
        config.STREET_LIGHTING: [{}, {lowest: Decimal("4.5")}, {}],
    }
    for defect_type, figures in stated.items():
        document_name = config.DEFECT_STANDARD_DOCUMENTS[defect_type]
        requirements = invented_standards["standards"][defect_type].get("requirements", [])
        written = invented_standards["invented"][defect_type]
        assert len(requirements) == len(figures), (
            document_name
            + " was read as "
            + str(len(requirements))
            + " requirements where it states "
            + str(len(figures))
            + "."
        )
        for requirement, wanted, stated_requirement in zip(requirements, figures, written,
                                                           strict=True):
            sentence = requirement.get("requirement_sentence")
            assert sentence == stated_requirement[2], (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " keeps the sentence "
                + repr(str(sentence)[:160])
                + " where the standard states "
                + repr(stated_requirement[2])
                + ". A requirement keeps the sentence it was read from, word for word, so the "
                "review and the inspection request quote the standard itself."
            )
            rules = requirement.get("rules", {})
            exact = isinstance(rules, dict) and all(isinstance(figure, Decimal)
                                                    for figure in rules.values())
            assert rules == wanted and exact, (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " was read with the figures "
                + repr(rules)
                + " where its sentence states "
                + repr(wanted)
                + ". Each requirement keeps its own figures, read from its own standard as exact "
                "decimals, and no others; none of these figures is Wrenford's, so none can come "
                "from anywhere but the standard read."
            )


def test_twin_depth_figures_read_from_own_documents(invented_standards):
    from decimal import Decimal

    from config import config
    from knowledge.standard_reading import read_defect_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    reordered = list(reversed(documents))
    for defect_type in config.DEFECT_STANDARD_DOCUMENTS:
        again = read_defect_standard(defect_type, reordered)
        assert again == invented_standards["standards"][defect_type], (
            "The standard of "
            + defect_type
            + " reads differently when the documents are handed over in another order. A "
            "standard is found by its document's name, wherever it stands among the documents."
        )
    footway_document = config.DEFECT_STANDARD_DOCUMENTS[config.FOOTWAY]
    amended = []
    for document in documents:
        document_text = document["document_text"]
        if document["document_name"] == footway_document:
            document_text = document_text.replace("vertical step of at least 15 millimetres",
                                                  "vertical step of at least 16 millimetres")
        amended.append({"document_name": document["document_name"],
                        "document_text": document_text})
    amended_figures = {}
    for defect_type in [config.CARRIAGEWAY, config.FOOTWAY]:
        amended_figures[defect_type] = [
            requirement.get("rules", {}).get(config.MINIMUM_RULE)
            for requirement in read_defect_standard(defect_type, amended).get("requirements", [])
            if requirement.get("requirement_kind") == config.DEFECT_DEPTH_REQUIREMENT]
    assert amended_figures == {config.CARRIAGEWAY: [Decimal("35")],
                               config.FOOTWAY: [Decimal("16")]}, (
        "With only the footway standard's depth figure changed to 16 millimetres, the two twins "
        "read "
        + str(amended_figures)
        + ". Changing one standard changes that defect type's figure and no other's, because "
        "each is read from its own document."
    )
    copied = documents + [dict(documents[0])]
    refusal = None
    try:
        read_defect_standard(config.CARRIAGEWAY, copied)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    carriageway_document = config.DEFECT_STANDARD_DOCUMENTS[config.CARRIAGEWAY]
    assert refusal is not None and carriageway_document in refusal, (
        "Reading the carriageway standard from documents holding two copies of it gave "
        + str(refusal)
        + ". Which copy states the requirements cannot be told, so it is refused with a "
        "ValueError naming the document."
    )


def test_band_and_thresholds_read_however_laid_out(invented_standards):
    from decimal import Decimal

    from config import config
    from knowledge.standard_reading import read_defect_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standard_documents = set(config.DEFECT_STANDARD_DOCUMENTS.values())
    relaid = []
    for document in invented_standards["documents"]:
        document_text = document["document_text"]
        if document["document_name"] in standard_documents:
            document_text = (document_text.replace(" is at least ", " is at\nleast  ")
                             .replace(" millimetres and at most ", " millimetres\nand at   most ")
                             .replace(" over a footway, or less than ",
                                      " over a\nfootway,  or less   than ")
                             .replace(" degrees from ", " degrees\nfrom  ")
                             .replace(" out from the kerb", "\nout from  the kerb"))
        relaid.append({"document_name": document["document_name"],
                       "document_text": document_text})
    differing = []
    for defect_type in config.DEFECT_STANDARD_DOCUMENTS:
        try:
            again = read_defect_standard(defect_type, relaid)
        except Exception as error:
            again = type(error).__name__ + ": " + str(error)
        if again != invented_standards["standards"][defect_type]:
            differing.append(defect_type + " read as " + repr(again)[:200])
    assert len(differing) == 0, (
        "With each standard's lines broken in other places and runs of spaces between its "
        "words, as text taken from a PDF can come, the standards read differently: "
        + str(differing[:2])
        + ". A standard is read from its sections, in which every run of spaces and line breaks "
        "is one space, so the same standard reads the same however its lines fall."
    )
    replacements = {
        config.CARRIAGEWAY: ("at least 120 millimetres and at most 1300 millimetres",
                             "at least 140 millimetres and at most 1250 millimetres"),
        config.DRAINAGE: ("at least 1.3 metres out", "at least 0.8 metres out"),
        config.VEGETATION: ("less than 2.2 metres over a footway, or less than 4.9 metres",
                            "less than 2.6 metres over a footway, or less than 5.3 metres"),
        config.STREET_LIGHTING: ("at least 4.5 degrees", "at least 6.5 degrees")}
    changed = []
    for document in invented_standards["documents"]:
        document_text = document["document_text"]
        for defect_type, (before, after) in replacements.items():
            if document["document_name"] == config.DEFECT_STANDARD_DOCUMENTS[defect_type]:
                document_text = document_text.replace(before, after)
        changed.append({"document_name": document["document_name"],
                        "document_text": document_text})
    lowest, highest = config.MINIMUM_RULE, config.MAXIMUM_RULE
    over_footway, over_carriageway = config.FOOTWAY_LIMIT_RULE, config.CARRIAGEWAY_LIMIT_RULE
    wanted = {
        config.CARRIAGEWAY: (config.DEFECT_WIDTH_REQUIREMENT,
                             {lowest: Decimal("140"), highest: Decimal("1250")}),
        config.FOOTWAY: (config.DEFECT_WIDTH_REQUIREMENT,
                         {lowest: Decimal("110"), highest: Decimal("1100")}),
        config.DRAINAGE: (config.WATER_REACH_REQUIREMENT, {lowest: Decimal("0.8")}),
        config.VEGETATION: (config.CLEAR_HEIGHT_REQUIREMENT,
                            {over_footway: Decimal("2.6"), over_carriageway: Decimal("5.3")}),
        config.STREET_LIGHTING: (config.COLUMN_LEAN_REQUIREMENT, {lowest: Decimal("6.5")})}
    got = {}
    for defect_type, (kind, _figures) in wanted.items():
        try:
            requirements = read_defect_standard(defect_type, changed).get("requirements", [])
            got[defect_type] = (kind, [requirement.get("rules") for requirement in requirements
                                       if requirement.get("requirement_kind") == kind])
        except Exception as error:
            got[defect_type] = (kind, type(error).__name__ + ": " + str(error)[:120])
    expected = {defect_type: (kind, [figures]) for defect_type, (kind, figures) in wanted.items()}
    assert got == expected, (
        "With the carriageway standard's width band changed to 140 to 1250 millimetres, the "
        "reach asked at 0.8 metres, the clear heights at 2.6 and 5.3 metres and the lean at 6.5 "
        "degrees, the standards read "
        + str(got)
        + " where they state "
        + str(expected)
        + ". The width band keeps both its figures, the lower as its minimum and the higher as "
        "its maximum; the clear height keeps its footway figure and its carriageway figure; "
        "each follow-on question keeps the figure its own standard states; and the footway "
        "standard's band, left alone, is unchanged."
    )


def test_unknown_type_refused_reading_repeatable(invented_standards):
    from config import config
    from knowledge.standard_reading import read_defect_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    refusal = None
    try:
        read_defect_standard("Road markings", documents)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "Road markings" in refusal, (
        "Reading the standard of a defect type with none configured gave "
        + str(refusal)
        + ". A defect type with no standard has no requirements to screen against, and is "
        "refused with a ValueError naming it."
    )
    for defect_type in config.DEFECT_STANDARD_DOCUMENTS:
        again = read_defect_standard(defect_type, documents)
        assert again == invented_standards["standards"][defect_type], (
            "The standard of "
            + defect_type
            + " reads differently the second time. A standard is read from its sections, the "
            "same way every time."
        )
    lighting_standard = config.DEFECT_STANDARD_DOCUMENTS[config.STREET_LIGHTING]
    refusal = None
    try:
        read_defect_standard(config.STREET_LIGHTING,
                             [document for document in documents
                              if document["document_name"] != lighting_standard])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and lighting_standard in refusal, (
        "Reading the street lighting standard from documents that do not include it gave "
        + str(refusal)
        + ". A standard missing from the documents read leaves its defect type with nothing to "
        "screen against, so it is refused with a ValueError naming the document."
    )
    fixed = ("Location fixed. The location of the defect is fixed to a house number, a lamp "
             "column number or a road junction.")
    depth = "Depth. The defect has a depth or a vertical step of at least 25 millimetres."
    photograph = "Photograph. A photograph of the defect is attached to the report."
    broken = {
        config.CARRIAGEWAY: (config.FOOTWAY, ["1 " + fixed, "2 " + depth]),
        config.FOOTWAY: (config.FOOTWAY, ["1 " + fixed, "2 " + depth,
                                          "3 " + depth.replace("25 millimetres",
                                                               "30 millimetres")]),
        config.DRAINAGE: (config.DRAINAGE, ["1 " + fixed, "2 " + photograph, "4 " + depth]),
        config.VEGETATION: (config.VEGETATION,
                            ["1 " + fixed, "2 Root spread. The roots spread at most 3 metres "
                                           "under the footway."]),
    }
    broken_documents = []
    for defect_type, (stated_type, requirement_lines) in broken.items():
        broken_documents.append({
            "document_name": config.DEFECT_STANDARD_DOCUMENTS[defect_type],
            "document_text": (
                "WRENFORD BOROUGH COUNCIL\nDocument Standard 98 | Issue 1 | Owner: Highways "
                "Asset Management | Review cycle: every twelve months\n1. Purpose and scope\nThis "
                "standard is laid out wrongly on purpose.\n2. What this standard covers\nThe "
                "defect type is "
                + stated_type
                + ".\n3. Who reports\nWhoever sees the defect.\n4. Requirements\nEvery "
                "requirement below must be Met, in this order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n"),
        })
    broken_documents.append({
        "document_name": config.DEFECT_STANDARD_DOCUMENTS[config.STREET_LIGHTING],
        "document_text": (
            "WRENFORD BOROUGH COUNCIL\nDocument Standard 98 | Issue 1 | Owner: Highways Asset "
            "Management | Review cycle: every twelve months\n1. Purpose and scope\nThis standard "
            "is laid out wrongly on purpose.\n2. What this standard covers\nThe defect type is "
            "Street lighting.\n3. Who reports\nWhoever sees the defect.\n"),
    })
    meanings = {
        config.CARRIAGEWAY: "states another defect type",
        config.FOOTWAY: "states the depth requirement twice",
        config.DRAINAGE: "numbers its requirements 1, 2 and 4",
        config.VEGETATION: "states a requirement the screening cannot read",
        config.STREET_LIGHTING: "has no requirements section",
    }
    for defect_type, meaning in meanings.items():
        document_name = config.DEFECT_STANDARD_DOCUMENTS[defect_type]
        refusal = None
        try:
            read_defect_standard(defect_type, broken_documents)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and document_name in refusal, (
            "A standard that "
            + meaning
            + " was read, giving "
            + str(refusal)
            + ". A standard that cannot be read as the Report Rules lay it out would screen "
            "reports against requirements nobody wrote, so it is refused with a ValueError "
            "naming the document."
        )


def test_requirements_met_at_boundary(live_files):
    from decimal import Decimal

    from config import config
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import screen_report

    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    templates = {}
    for report_reference in reports[config.REPORT_REFERENCE_COLUMN]:
        report = fetch_report(reports, report_reference)
        if (report[config.DEFECT_TYPE_COLUMN] not in templates
                and report[config.LINKED_DEFECT_TYPE_COLUMN] == config.NO_LINKED_DEFECT):
            templates[report[config.DEFECT_TYPE_COLUMN]] = report
    lowest, highest = config.MINIMUM_RULE, config.MAXIMUM_RULE
    cw, fw, dr = config.CARRIAGEWAY, config.FOOTWAY, config.DRAINAGE
    vg, sl = config.VEGETATION, config.STREET_LIGHTING
    invented = {
        cw: [(config.DEFECT_DEPTH_REQUIREMENT, "Depth", {lowest: Decimal("37")}),
             (config.DEFECT_WIDTH_REQUIREMENT, "Width",
              {lowest: Decimal("130"), highest: Decimal("1400")})],
        fw: [(config.DEFECT_DEPTH_REQUIREMENT, "Depth", {lowest: Decimal("18")})],
        dr: [(config.WATER_DEPTH_REQUIREMENT, "Water depth", {lowest: Decimal("55")}),
             (config.WATER_REACH_REQUIREMENT, "Water reach", {lowest: Decimal("1.4")})],
        vg: [(config.CLEAR_HEIGHT_REQUIREMENT, "Clear height",
              {config.FOOTWAY_LIMIT_RULE: Decimal("2.3"),
               config.CARRIAGEWAY_LIMIT_RULE: Decimal("5.2")})],
        sl: [(config.COLUMN_LEAN_REQUIREMENT, "Column lean", {lowest: Decimal("5.5")})],
    }
    depth, width, water = "defect_depth_mm", "defect_width_mm", "water_depth_mm"
    reach, location = config.WATER_REACH_COLUMN, config.WATER_LOCATION_COLUMN
    clear, overhang = config.CLEAR_HEIGHT_COLUMN, config.OVERHANG_LOCATION_COLUMN
    lean, fault = config.COLUMN_LEAN_COLUMN, config.LIGHTING_FAULT_COLUMN
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    road, path = config.CARRIAGEWAY_PART, config.FOOTWAY_PART
    cases = [
        (cw, "Depth", {depth: 37}, met), (cw, "Depth", {depth: 36}, not_met),
        (cw, "Width", {width: 130}, met), (cw, "Width", {width: 129}, not_met),
        (cw, "Width", {width: 1400}, met), (cw, "Width", {width: 1401}, not_met),
        (fw, "Depth", {depth: 18}, met), (fw, "Depth", {depth: 17}, not_met),
        (dr, "Water depth", {water: 55}, met), (dr, "Water depth", {water: 54}, not_met),
        (dr, "Water reach", {location: road, reach: 1.4}, met),
        (dr, "Water reach", {location: road, reach: 1.3}, not_met),
        (vg, "Clear height", {overhang: path, clear: 2.2}, met),
        (vg, "Clear height", {overhang: path, clear: 2.3}, not_met),
        (vg, "Clear height", {overhang: road, clear: 5.1}, met),
        (vg, "Clear height", {overhang: road, clear: 5.2}, not_met),
        (sl, "Column lean", {fault: config.LEANING_COLUMN, lean: 5.5}, met),
        (sl, "Column lean", {fault: config.LEANING_COLUMN, lean: 5.4}, not_met),
    ]
    assert set(invented) <= set(templates), (
        "The live queue holds unlinked reports of the defect types "
        + str(sorted(templates))
        + ", where it holds one of every defect type."
    )
    wrong = []
    for defect_type, requirement_name, answers, expected in cases:
        standard = {"defect_type": defect_type,
                    "document_name": "Invented-Boundary-Standard.pdf",
                    "requirements": [{"requirement_kind": kind, "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "rules": rules}
                                     for kind, name, rules in invented[defect_type]]}
        report = dict(templates[defect_type])
        report.update({config.REPORT_REFERENCE_COLUMN: "SC0950101",
                       config.LIVE_STREET_WORKS_COLUMN: 0,
                       config.CONFLICTING_STREET_WORKS_COLUMN: 0})
        report.update(answers)
        screening = screen_report(report, {defect_type: standard})
        assert isinstance(screening, dict) and isinstance(screening.get("requirements"), list), (
            "screen_report returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each requirement."
        )
        results = {requirement.get("requirement_name"): requirement.get("result")
                   for requirement in screening["requirements"] if isinstance(requirement, dict)}
        if results.get(requirement_name) != expected:
            wrong.append(defect_type + " " + requirement_name + " with " + str(answers) + " is "
                         + str(results.get(requirement_name)) + ", not " + expected)
    assert len(wrong) == 0, (
        "Under standards whose figures no Wrenford standard uses, the screening misplaces "
        "boundaries: "
        + "; ".join(wrong[:5])
        + ". A requirement written at least is met at its figure, one written at least and at "
        "most at either of its figures, and one written less than is not met at its figure, "
        "every reading compared as an exact decimal - a reach of 1.4 against at least 1.4 is "
        "met, and a clear height of exactly 2.3 against less than 2.3 leaves the street clear - "
        "and every figure comes from the standard handed over, never from a figure written into "
        "the code."
    )


def test_follow_on_questions_ask_only_answers_needed(live_files):
    from decimal import Decimal

    from config import config
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import screen_report

    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    template = None
    for report_reference in reports[config.REPORT_REFERENCE_COLUMN]:
        report = fetch_report(reports, report_reference)
        if template is None and report[config.DEFECT_TYPE_COLUMN] == config.DRAINAGE:
            template = report
    assert template is not None, "The live queue holds no drainage report."
    standard = {"defect_type": config.DRAINAGE,
                "document_name": "Invented-Follow-On-Standard.pdf",
                "requirements": []}
    for kind, name, rules in [
            (config.WATER_REACH_REQUIREMENT, "Water reach", {config.MINIMUM_RULE: Decimal("1.4")}),
            (config.CLEAR_HEIGHT_REQUIREMENT, "Clear height",
             {config.FOOTWAY_LIMIT_RULE: Decimal("2.3"),
              config.CARRIAGEWAY_LIMIT_RULE: Decimal("5.2")}),
            (config.COLUMN_LEAN_REQUIREMENT, "Column lean", {config.MINIMUM_RULE: Decimal("5.5")}),
            (config.LOCATION_FIXED_REQUIREMENT, "Location fixed", {}),
            (config.DEFECT_DEPTH_REQUIREMENT, "Depth", {config.MINIMUM_RULE: Decimal("37")}),
            (config.DEFECT_WIDTH_REQUIREMENT, "Width",
             {config.MINIMUM_RULE: Decimal("130"), config.MAXIMUM_RULE: Decimal("1400")})]:
        standard["requirements"].append({"requirement_kind": kind, "requirement_name": name,
                                         "requirement_sentence": ("An invented requirement: "
                                                                  + name + "."),
                                         "rules": rules})
    reach, location = config.WATER_REACH_COLUMN, config.WATER_LOCATION_COLUMN
    clear, overhang = config.CLEAR_HEIGHT_COLUMN, config.OVERHANG_LOCATION_COLUMN
    lean, fault = config.COLUMN_LEAN_COLUMN, config.LIGHTING_FAULT_COLUMN
    fixed, depth, width = "location_fixed", "defect_depth_mm", "defect_width_mm"
    road, path = config.CARRIAGEWAY_PART, config.FOOTWAY_PART
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    cases = [
        ("Water reach", {location: path, reach: None}, met, []),
        ("Water reach", {location: path, reach: 0.2}, met, []),
        ("Water reach", {location: road, reach: 1.4}, met, []),
        ("Water reach", {location: road, reach: 1.3}, not_met, []),
        ("Water reach", {location: road, reach: None}, undocumented, [reach]),
        ("Water reach", {location: None, reach: 2.0}, undocumented, [location]),
        ("Water reach", {location: None, reach: None}, undocumented, [location]),
        ("Clear height", {overhang: path, clear: 2.2}, met, []),
        ("Clear height", {overhang: path, clear: 2.3}, not_met, []),
        ("Clear height", {overhang: road, clear: 2.3}, met, []),
        ("Clear height", {overhang: road, clear: 5.2}, not_met, []),
        ("Clear height", {overhang: path, clear: None}, undocumented, [clear]),
        ("Clear height", {overhang: None, clear: 2.0}, undocumented, [overhang]),
        ("Clear height", {overhang: None, clear: None}, undocumented, [overhang]),
        ("Column lean", {fault: config.LAMP_OUT, lean: None}, met, []),
        ("Column lean", {fault: config.DAY_BURNING, lean: 0.0}, met, []),
        ("Column lean", {fault: config.LEANING_COLUMN, lean: 5.5}, met, []),
        ("Column lean", {fault: config.LEANING_COLUMN, lean: 5.4}, not_met, []),
        ("Column lean", {fault: config.LEANING_COLUMN, lean: None}, undocumented, [lean]),
        ("Column lean", {fault: None, lean: 7.0}, undocumented, [fault]),
        ("Location fixed", {fixed: answered_yes}, met, []),
        ("Location fixed", {fixed: answered_no}, not_met, []),
        ("Location fixed", {fixed: None}, undocumented, [fixed]),
        ("Depth", {depth: None}, undocumented, [depth]),
        ("Width", {width: None}, undocumented, [width]),
    ]
    wrong = []
    for requirement_name, answers, expected, expected_blanks in cases:
        report = dict(template)
        report.update({config.REPORT_REFERENCE_COLUMN: "SC0950102",
                       config.LINKED_DEFECT_TYPE_COLUMN: config.NO_LINKED_DEFECT,
                       config.LIVE_STREET_WORKS_COLUMN: 0,
                       config.CONFLICTING_STREET_WORKS_COLUMN: 0})
        report.update(answers)
        screening = screen_report(report, {config.DRAINAGE: standard})
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_report returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each of the standard's "
            "requirements, in the standard's order."
        )
        found = [requirement for requirement in screening["requirements"]
                 if requirement.get("requirement_name") == requirement_name]
        if (len(found) != 1 or found[0].get("result") != expected
                or list(found[0].get("blank_answers", [])) != expected_blanks):
            wrong.append(requirement_name + " with " + str(answers) + " gives "
                         + str([(requirement.get("result"), requirement.get("blank_answers"))
                                for requirement in found])
                         + ", not " + expected + " asking for " + str(expected_blanks))
    assert len(wrong) == 0, (
        "The follow-on questions are screened wrongly: "
        + "; ".join(wrong[:4])
        + ". The water reach is asked only where the water stands on the carriageway, the clear "
        "height is held to the standard's footway figure over a footway and its carriageway "
        "figure over a carriageway, and the column lean is asked only of a leaning column: a "
        "reach on the footway and the lean of a lamp that is out or burns by day are Met "
        "whatever else the report says; otherwise Met at or beyond the figure, Not met short of "
        "it, and Not documented, asking for the follow-on answer alone, when it is blank; and "
        "Not documented, asking for the first answer alone, when that is blank. A yes-or-no "
        "requirement or a reading left blank asks for that answer."
    )


def test_linked_report_screened_under_both_standards(live_files):
    from decimal import Decimal

    from config import config
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import screen_report

    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    template = fetch_report(reports, reports[config.REPORT_REFERENCE_COLUMN].iloc[0])
    cw, fw = config.CARRIAGEWAY, config.FOOTWAY
    standards = {
        cw: {"defect_type": cw, "document_name": "Invented-Carriageway-Standard.pdf",
             "requirements": [
                 {"requirement_kind": config.LOCATION_FIXED_REQUIREMENT,
                  "requirement_name": "Location fixed",
                  "requirement_sentence": ("The location of the defect is fixed to a house "
                                           "number, a lamp column number or a road junction."),
                  "rules": {}},
                 {"requirement_kind": config.DEFECT_DEPTH_REQUIREMENT,
                  "requirement_name": "Depth",
                  "requirement_sentence": ("The defect has a depth or a vertical step of at "
                                           "least 45 millimetres."),
                  "rules": {config.MINIMUM_RULE: Decimal("45")}},
                 {"requirement_kind": config.DEFECT_WIDTH_REQUIREMENT, "requirement_name": "Width",
                  "requirement_sentence": ("The defect measures at least 160 millimetres and at "
                                           "most 1600 millimetres across."),
                  "rules": {config.MINIMUM_RULE: Decimal("160"),
                            config.MAXIMUM_RULE: Decimal("1600")}}]},
        fw: {"defect_type": fw, "document_name": "Invented-Footway-Standard.pdf",
             "requirements": [
                 {"requirement_kind": config.DEFECT_DEPTH_REQUIREMENT,
                  "requirement_name": "Depth",
                  "requirement_sentence": ("The defect has a depth or a vertical step of at "
                                           "least 25 millimetres."),
                  "rules": {config.MINIMUM_RULE: Decimal("25")}},
                 {"requirement_kind": config.DEFECT_WIDTH_REQUIREMENT, "requirement_name": "Width",
                  "requirement_sentence": ("The defect measures at least 120 millimetres and at "
                                           "most 1200 millimetres across."),
                  "rules": {config.MINIMUM_RULE: Decimal("120"),
                            config.MAXIMUM_RULE: Decimal("1200")}},
                 {"requirement_kind": config.PHOTOGRAPH_REQUIREMENT,
                  "requirement_name": "Photograph",
                  "requirement_sentence": "A photograph of the defect is attached to the report.",
                  "rules": {}}]}}
    answers = {config.REPORT_REFERENCE_COLUMN: "SC0950103",
               config.STREET_CATEGORY_COLUMN: config.LOCAL_ROAD,
               config.LIVE_STREET_WORKS_COLUMN: 0,
               config.CONFLICTING_STREET_WORKS_COLUMN: 0,
               "location_fixed": config.BOOLEAN_YES, "defect_depth_mm": 35,
               "defect_width_mm": 300, "photograph_attached": config.BOOLEAN_YES}
    cw_document = standards[cw]["document_name"]
    fw_document = standards[fw]["document_name"]
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    cases = [
        ("a footway report linked to carriageway", fw, cw, [fw, cw],
         [(fw, fw_document, "Depth", met), (fw, fw_document, "Width", met),
          (fw, fw_document, "Photograph", met), (cw, cw_document, "Location fixed", met),
          (cw, cw_document, "Depth", not_met), (cw, cw_document, "Width", met)],
         [5, 1, 0], config.DECISION_INSPECT),
        ("a carriageway report linked to footway", cw, fw, [cw, fw],
         [(cw, cw_document, "Location fixed", met), (cw, cw_document, "Depth", not_met),
          (cw, cw_document, "Width", met), (fw, fw_document, "Depth", met),
          (fw, fw_document, "Width", met), (fw, fw_document, "Photograph", met)],
         [5, 1, 0], config.DECISION_INSPECT),
        ("the same footway report with no linked defect", fw, config.NO_LINKED_DEFECT, [fw],
         [(fw, fw_document, "Depth", met), (fw, fw_document, "Width", met),
          (fw, fw_document, "Photograph", met)],
         [3, 0, 0], config.DECISION_ORDER)]
    for meaning, defect_type, linked_type, defect_types, expected, counts, assessment in cases:
        report = dict(template, **answers)
        report[config.DEFECT_TYPE_COLUMN] = defect_type
        report[config.LINKED_DEFECT_TYPE_COLUMN] = linked_type
        screening = screen_report(report, standards)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_report returned "
            + str(screening)[:200]
            + " for "
            + meaning
            + ", where it returns the screening with one result for each requirement."
        )
        got = [(requirement.get("defect_type"), requirement.get("standard_document"),
                requirement.get("requirement_name"), requirement.get("result"))
               for requirement in screening["requirements"]]
        got_counts = [screening.get(column_name)
                      for column_name in config.REQUIREMENT_COUNT_RESULTS]
        assert (got == expected and got_counts == counts
                and screening.get("assessment") == assessment
                and list(screening.get("defect_types", [])) == defect_types
                and screening.get("report_reference") == "SC0950103"), (
            "For "
            + meaning
            + " 35 millimetres deep, the screening gives "
            + str(got)
            + " with counts "
            + str(got_counts)
            + ", the assessment "
            + str(screening.get("assessment"))
            + " and the defect types "
            + str(screening.get("defect_types"))
            + ", where the Report Rules give "
            + str(expected)
            + ", "
            + str(counts)
            + ", "
            + assessment
            + " and "
            + str(defect_types)
            + ". A linked report is screened under its own standard first and then under its "
            "linked type's, each requirement keeping the defect type and the document it came "
            "from: 35 millimetres passes the footway standard's 25 and fails the carriageway "
            "standard's 45, so the report goes for inspection on site."
        )


def test_coordination_triggers_follow_register_and_rules(live_files, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.register_cleaning import clean_register_records
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import count_live_street_works, find_coordination_triggers

    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    template = fetch_report(reports, reports[config.REPORT_REFERENCE_COLUMN].iloc[0])
    raw_register = load_raw_dataset(config.STREET_WORKS_REGISTER_FILE)
    written = [("SWP0051301", "mill lane", "Carriageway", "water company", "Water main repair",
                "Live"),
               ("SWP0051302", "MILL LANE", "FOOTWAY", "Gas network", "Gas leak repair", "LIVE"),
               ("SWP0051303", "Mill Lane", "carriageway", "Telecoms", "Duct repair", "live"),
               ("SWP0051304", "Mill Lane", "Footway", "Electricity network", "Cable fault",
                "Live"),
               ("SWP0051305", "Mill Lane", "Carriageway", "Council contractor", "Resurfacing",
                "CANCELLED"),
               ("SWP0051306", "Kiln Lane", "Carriageway", "Water company", "Valve repair",
                "Live")]
    rows = pd.concat([raw_register.head(1)] * len(written), ignore_index=True)
    for position, values in enumerate(written):
        for column_name, value in zip([config.PERMIT_REFERENCE_COLUMN, config.STREET_NAME_COLUMN,
                                       config.WORKS_LOCATION_COLUMN, "works_promoter",
                                       "works_description", config.PERMIT_STATUS_COLUMN],
                                      values, strict=True):
            rows.loc[position, column_name] = value
    register_path = str(tmp_path / "planted_register.csv")
    rows.to_csv(register_path, index=False)
    try:
        register = clean_register_records(load_raw_dataset(register_path))
    except Exception as error:
        register = error
    assert isinstance(register, pd.DataFrame) and len(register) == len(written), (
        "clean_register_records gave "
        + repr(register)[:200]
        + " for six planted permits written in other capitals, where it returns them cleaned."
    )
    cw, fw, dr = config.CARRIAGEWAY, config.FOOTWAY, config.DRAINAGE
    vg, sl, unlinked = config.VEGETATION, config.STREET_LIGHTING, config.NO_LINKED_DEFECT
    first, second, third, fourth = "SWP0051301", "SWP0051302", "SWP0051303", "SWP0051304"
    count_cases = [
        ("a carriageway report on Mill Lane", cw, unlinked, "Mill Lane",
         [first, second, third, fourth], [first, third]),
        ("a vegetation report linked to footway on Mill Lane", vg, fw, "Mill Lane",
         [first, second, third, fourth], [second, fourth]),
        ("a vegetation report on Mill Lane", vg, unlinked, "Mill Lane",
         [first, second, third, fourth], []),
        ("a street lighting report on Mill Lane", sl, unlinked, "Mill Lane",
         [first, second, third, fourth], [second, fourth]),
        ("a drainage report linked to street lighting on Mill Lane", dr, sl, "Mill Lane",
         [first, second, third, fourth], [first, second, third, fourth]),
        ("a footway report on Kiln Lane", fw, unlinked, "Kiln Lane", ["SWP0051306"], [])]
    wrong = []
    for meaning, defect_type, linked_type, street_name, live, conflicting in count_cases:
        report = dict(template)
        report.update({config.REPORT_REFERENCE_COLUMN: "SC0950104",
                       config.DEFECT_TYPE_COLUMN: defect_type,
                       config.LINKED_DEFECT_TYPE_COLUMN: linked_type,
                       config.STREET_NAME_COLUMN: street_name})
        counted = count_live_street_works(register, report)
        expected = {config.LIVE_STREET_WORKS_COLUMN: len(live),
                    config.CONFLICTING_STREET_WORKS_COLUMN: len(conflicting),
                    "live_permits": live, "conflicting_permits": conflicting}
        got = ({key: counted.get(key) for key in expected} if isinstance(counted, dict)
               else counted)
        if got != expected:
            wrong.append(meaning + " counts " + str(got) + ", not " + str(expected))
    assert len(wrong) == 0, (
        "The street works permits live on a report's street are counted wrongly from a register "
        "whose streets, works locations and statuses are written in other capitals: "
        + "; ".join(wrong[:3])
        + ". Every permit live on the report's street is counted, in the order of its reference, "
        "and those whose works are in the same part of the highway as the report's defect type "
        "or its linked type - a carriageway or drainage defect in the carriageway, a footway "
        "defect or a street light in the footway, vegetation in neither - are its conflicting "
        "permits; a cancelled permit is never counted, and a permit on another street is not "
        "beside the defect."
    )
    refusal = None
    try:
        blank_street = dict(template)
        blank_street.update({config.REPORT_REFERENCE_COLUMN: "SC0950105",
                             config.STREET_NAME_COLUMN: None})
        count_live_street_works(register, blank_street)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "SC0950105" in refusal, (
        "Counting the permits beside a report that gives no street gave "
        + str(refusal)
        + ". With no street there is no register to count from, so it is refused with a "
        "ValueError naming the report."
    )
    live_works = config.LIVE_STREET_WORKS_TRIGGER
    sensitive, high = config.TRAFFIC_SENSITIVE_TRIGGER, config.HIGH_COLUMN_TRIGGER
    local, busy = config.LOCAL_ROAD, config.TRAFFIC_SENSITIVE
    trigger_cases = [
        (cw, unlinked, local, None, 0, []),
        (cw, unlinked, local, None, 1, [live_works]),
        (cw, unlinked, busy, None, 0, [sensitive]),
        (cw, unlinked, None, None, 0, [sensitive]),
        (fw, unlinked, busy, None, 0, []),
        (dr, unlinked, busy, None, 0, [sensitive]),
        (vg, cw, busy, None, 0, [sensitive]),
        (sl, unlinked, local, 8.0, 0, []),
        (sl, unlinked, local, 8.1, 0, [high]),
        (sl, unlinked, local, None, 0, [high]),
        (fw, sl, local, 12.0, 0, [high]),
        (cw, unlinked, local, 12.0, 0, []),
        (dr, sl, None, None, 2, [live_works, sensitive, high])]
    wrong = []
    for defect_type, linked_type, category, height, conflicts, expected in trigger_cases:
        report = dict(template)
        report.update({config.REPORT_REFERENCE_COLUMN: "SC0950106",
                       config.DEFECT_TYPE_COLUMN: defect_type,
                       config.LINKED_DEFECT_TYPE_COLUMN: linked_type,
                       config.STREET_CATEGORY_COLUMN: category,
                       config.COLUMN_HEIGHT_COLUMN: height,
                       config.LIVE_STREET_WORKS_COLUMN: conflicts + 1,
                       config.CONFLICTING_STREET_WORKS_COLUMN: conflicts})
        triggers = find_coordination_triggers(report)
        if not isinstance(triggers, list) or triggers != expected:
            wrong.append(defect_type + " linked to " + linked_type + " on a street "
                         + str(category) + " with a column of " + str(height) + " m beside "
                         + str(conflicts) + " conflicting permits meets " + str(triggers)
                         + ", not " + str(expected))
    assert len(wrong) == 0, (
        "The coordination triggers are found wrongly: "
        + "; ".join(wrong[:4])
        + ". A conflicting live permit triggers; so does a carriageway or drainage defect, as "
        "the defect type or the linked type, on a street that is not a local road; and a street "
        "light, either way, on a column taller than 8 metres. A blank street category counts as "
        "traffic sensitive and a blank column height as taller than 8 metres, and the triggers "
        "are listed in that order."
    )


def test_assessment_decisions_and_worked_example(live_files, tmp_path):
    from decimal import Decimal

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.report_cleaning import clean_report_records
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import screen_report

    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    template = fetch_report(reports, reports[config.REPORT_REFERENCE_COLUMN].iloc[0])
    cw, sl = config.CARRIAGEWAY, config.STREET_LIGHTING
    standards = {
        cw: {"defect_type": cw, "document_name": "Invented-Assessment-Standard.pdf",
             "requirements": [
                 {"requirement_kind": config.LOCATION_FIXED_REQUIREMENT,
                  "requirement_name": "Location fixed",
                  "requirement_sentence": ("The location of the defect is fixed to a house "
                                           "number, a lamp column number or a road junction."),
                  "rules": {}},
                 {"requirement_kind": config.DEFECT_DEPTH_REQUIREMENT,
                  "requirement_name": "Depth",
                  "requirement_sentence": ("The defect has a depth or a vertical step of at "
                                           "least 30 millimetres."),
                  "rules": {config.MINIMUM_RULE: Decimal("30")}}]},
        sl: {"defect_type": sl, "document_name": "Invented-Assessment-Standard.pdf",
             "requirements": [
                 {"requirement_kind": config.COLUMN_NUMBER_REQUIREMENT,
                  "requirement_name": "Column number",
                  "requirement_sentence": "The lamp column number is given on the report.",
                  "rules": {}},
                 {"requirement_kind": config.COLUMN_LEAN_REQUIREMENT,
                  "requirement_name": "Column lean",
                  "requirement_sentence": ("Where the column leans, it leans at least 5.5 degrees "
                                           "from upright."),
                  "rules": {config.MINIMUM_RULE: Decimal("5.5")}}]}}
    fixed, depth = "location_fixed", "defect_depth_mm"
    category, conflicts = config.STREET_CATEGORY_COLUMN, config.CONFLICTING_STREET_WORKS_COLUMN
    fault, lean, height = (config.LIGHTING_FAULT_COLUMN, config.COLUMN_LEAN_COLUMN,
                           config.COLUMN_HEIGHT_COLUMN)
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    order, inspect, refer = config.DECISION_ORDER, config.DECISION_INSPECT, config.DECISION_REFER
    live_works = config.LIVE_STREET_WORKS_TRIGGER
    sensitive, high = config.TRAFFIC_SENSITIVE_TRIGGER, config.HIGH_COLUMN_TRIGGER
    base = {config.LINKED_DEFECT_TYPE_COLUMN: config.NO_LINKED_DEFECT, fixed: answered_yes,
            depth: 50, category: config.LOCAL_ROAD, config.LIVE_STREET_WORKS_COLUMN: 0,
            conflicts: 0, "column_number_given": answered_yes, fault: config.LAMP_OUT,
            lean: None, height: 6.0}
    cases = [
        (cw, {}, order, [2, 0, 0], []),
        (cw, {conflicts: 1, config.LIVE_STREET_WORKS_COLUMN: 1}, refer, [2, 0, 0], [live_works]),
        (cw, {category: config.TRAFFIC_SENSITIVE}, refer, [2, 0, 0], [sensitive]),
        (cw, {fixed: answered_no}, inspect, [1, 1, 0], []),
        (cw, {depth: 20}, inspect, [1, 1, 0], []),
        (cw, {fixed: None, category: config.TRAFFIC_SENSITIVE}, inspect, [1, 0, 1], [sensitive]),
        (sl, {}, order, [2, 0, 0], []),
        (sl, {fault: config.LEANING_COLUMN, lean: 7.0, height: 9.5}, refer, [2, 0, 0], [high]),
        (sl, {fault: config.LEANING_COLUMN, lean: None, height: None}, inspect, [1, 0, 1],
         [high]),
    ]
    for defect_type, answers, expected_assessment, expected_counts, expected_triggers in cases:
        report = dict(template)
        report[config.REPORT_REFERENCE_COLUMN] = "SC0950107"
        report[config.DEFECT_TYPE_COLUMN] = defect_type
        report.update(base)
        report.update(answers)
        screening = screen_report(report, standards)
        assert isinstance(screening, dict), (
            "screen_report returned "
            + type(screening).__name__
            + ", where it returns the screening of the report."
        )
        counts = [screening.get(column_name) for column_name in config.REQUIREMENT_COUNT_RESULTS]
        assert (screening.get("assessment") == expected_assessment
                and counts == expected_counts
                and list(screening.get("coordination_triggers", [])) == expected_triggers
                and list(screening.get("decisions_open", []))
                == config.DECISIONS_OPEN[expected_assessment]
                and screening.get("report_reference") == "SC0950107"), (
            "A "
            + defect_type.lower()
            + " report with "
            + str(answers)
            + " on a report that otherwise meets every requirement and triggers nothing is "
            "assessed "
            + str(screening.get("assessment"))
            + " with counts "
            + str(counts)
            + ", the coordination triggers "
            + str(screening.get("coordination_triggers"))
            + " and the decisions open "
            + str(screening.get("decisions_open"))
            + ", where the Highways Policy gives "
            + expected_assessment
            + ", "
            + str(expected_counts)
            + ", "
            + str(expected_triggers)
            + " and "
            + str(config.DECISIONS_OPEN[expected_assessment])
            + ". Any requirement Not met or Not documented makes the assessment INSPECT ON SITE "
            "before any coordination trigger counts, though the triggers are still found; "
            "otherwise a trigger makes it REFER FOR COORDINATION; ORDER REPAIR alone leaves every "
            "decision open."
        )
    written = live_files["raw_reports"].head(1).copy()
    for column_name, value in [(config.REPORT_REFERENCE_COLUMN, "52999"),
                               (config.REVISION_NUMBER_COLUMN, "1"),
                               ("report_channel", "Contact centre"),
                               (config.DEFECT_TYPE_COLUMN, "CW"),
                               (config.LINKED_DEFECT_TYPE_COLUMN, "DR")]:
        written[column_name] = value
    worked_standards = {
        cw: {"defect_type": cw, "document_name": "Worked-Example-Standard.pdf",
             "requirements": [
                 {"requirement_kind": config.DEFECT_DEPTH_REQUIREMENT,
                  "requirement_name": "Depth",
                  "requirement_sentence": ("The defect has a depth or a vertical step of at "
                                           "least 30 millimetres."),
                  "rules": {config.MINIMUM_RULE: Decimal("30")}},
                 {"requirement_kind": config.DEFECT_WIDTH_REQUIREMENT,
                  "requirement_name": "Width",
                  "requirement_sentence": ("The defect measures at least 100 millimetres and at "
                                           "most 1000 millimetres across."),
                  "rules": {config.MINIMUM_RULE: Decimal("100"),
                            config.MAXIMUM_RULE: Decimal("1000")}}]},
        config.DRAINAGE: {"defect_type": config.DRAINAGE,
                          "document_name": "Worked-Example-Standard.pdf",
                          "requirements": [
                              {"requirement_kind": config.WATER_DEPTH_REQUIREMENT,
                               "requirement_name": "Water depth",
                               "requirement_sentence": ("The standing water is at least 40 "
                                                        "millimetres deep."),
                               "rules": {config.MINIMUM_RULE: Decimal("40")}},
                              {"requirement_kind": config.WATER_REACH_REQUIREMENT,
                               "requirement_name": "Water reach",
                               "requirement_sentence": ("Where the water stands on the "
                                                        "carriageway, it reaches at least 1.5 "
                                                        "metres out from the kerb."),
                               "rules": {config.MINIMUM_RULE: Decimal("1.5")}}]}}
    width, water = "defect_width_mm", "water_depth_mm"
    location, reach = config.WATER_LOCATION_COLUMN, config.WATER_REACH_COLUMN
    outcomes = {}
    for label, sent in [("as sent", {}),
                        ("with the width written 3.95 in, the standing water 1.6 in and the "
                         "water on the PATH",
                         {width: "3.95 in", water: "1.6 in", location: "PATH"}),
                        ("with the street written LR, the width 3.95 in, the standing water "
                         "1.6 in and the reach 4.9 ft",
                         {category: "LR", width: "3.95 in", water: "1.6 in", reach: "4.9 ft"})]:
        variant = written.copy()
        for column_name, value in [(category, "TS"), (depth, "1.25 in"), (width, "3.9 in"),
                                   (water, "1.5 in"), (location, "ROAD"),
                                   (reach, "caller unsure")]:
            variant[column_name] = sent.get(column_name, value)
        sent_path = str(tmp_path / "worked_example.csv")
        variant.to_csv(sent_path, index=False)
        cleaned = clean_report_records(load_raw_dataset(sent_path))
        assert isinstance(cleaned, pd.DataFrame) and "SC0052999" in set(
                cleaned.get(config.REPORT_REFERENCE_COLUMN, [])), (
            "clean_report_records returned "
            + str(cleaned)[:200]
            + " for the worked example written 52999, where it returns the cleaned report as "
            "SC0052999."
        )
        cleaned_report = fetch_report(cleaned, "SC0052999")
        cleaned_report.update({config.LIVE_STREET_WORKS_COLUMN: 0, conflicts: 0})
        screening = screen_report(cleaned_report, worked_standards)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_report returned "
            + str(screening)[:200]
            + " for the worked example, where it returns its screening."
        )
        outcomes[label] = (screening.get("assessment"),
                           [requirement.get("result") for requirement in screening["requirements"]],
                           [list(requirement.get("blank_answers", []))
                            for requirement in screening["requirements"]],
                           list(screening.get("coordination_triggers", [])),
                           cleaned_report[depth], cleaned_report[width], cleaned_report[water],
                           cleaned_report[reach])
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    expected_outcomes = {
        "as sent": (inspect, [met, not_met, not_met, undocumented], [[], [], [], [reach]],
                    [sensitive], 32, 99, 38, None),
        "with the width written 3.95 in, the standing water 1.6 in and the water on the PATH": (
            refer, [met, met, met, met], [[], [], [], []], [sensitive], 32, 100, 41, None),
        "with the street written LR, the width 3.95 in, the standing water 1.6 in and the "
        "reach 4.9 ft": (order, [met, met, met, met], [[], [], [], []], [], 32, 100, 41, 1.5),
    }
    assert outcomes == expected_outcomes, (
        "The worked example of the Report Rules, section 13, screens as "
        + str(outcomes)
        + " where the Rules give "
        + str(expected_outcomes)
        + ": a depth of 1.25 in is 31.75, rounded half up once to 32 mm, at least 30, so Met; a "
        "width of 3.9 in is 99 mm, under 100, so Not met; standing water of 1.5 in is 38 mm, "
        "under 40, so Not met; the water stands on the road and its reach is written caller "
        "unsure, which is blank, so Not documented, asking for the reach alone; and the street "
        "written TS is traffic sensitive, so the trigger applies too, but the requirements come "
        "first. With the width at 3.95 in, 100 mm, the water at 1.6 in, 41 mm, and the water on "
        "the footway, every requirement is Met and the trigger makes it a referral; on a local "
        "road with a reach of 4.9 ft, 1.5 metres, nothing triggers and the repair may be ordered."
    )


def test_agents_called_once_in_order_with_tokens(saved_run):
    from config import config
    from persistence.database import session_factory
    from persistence.models import LlmCall

    assert saved_run["problem"] is None, saved_run["problem"]
    listed = saved_run["calls_at_gate"]
    assert (listed["status"] == 200 and isinstance(listed["body"], dict)
            and {"calls", "call_count", "expected_call_count"} <= set(listed["body"])
            and all("agent_name" in call for call in listed["body"]["calls"])), (
        "Reading the run's model calls at the pause returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit trail lists every call under the agent that made it, with their "
        "count and the count expected."
    )
    at_gate = listed["body"]
    reviewing = [config.AGENT_NAME_PLANNER, config.AGENT_NAME_REQUIREMENTS,
                 config.AGENT_NAME_EVIDENCE, config.AGENT_NAME_RISK, config.AGENT_NAME_WRITER]
    called = [call["agent_name"] for call in at_gate["calls"]]
    assert at_gate["call_count"] == len(at_gate["calls"]) == config.LLM_CALLS_BEFORE_DECISION, (
        "Reaching a Highways Inspector recorded a call count of "
        + str(at_gate["call_count"])
        + " and listed "
        + str(len(at_gate["calls"]))
        + " language model calls where the workflow allows "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ", one per reviewing agent. More means an agent ran twice; fewer, that one never ran."
    )
    assert called == reviewing, (
        "The agents ran in the order "
        + str(called)
        + ". Each one reads what the one before it concluded: the requirements are explained "
        "against the plan, the documents are reviewed against it, the figures are weighed with "
        "the plan in hand, and the recommendation draws on all of them."
    )
    assert at_gate["expected_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "Before any decision the service expected "
        + str(at_gate["expected_call_count"])
        + " calls. Until a decision is taken only the reviewing agents' calls are expected."
    )
    assert saved_run["at_gate"].get("llm_call_count") == config.LLM_CALLS_BEFORE_DECISION, (
        "The review record at the pause counts "
        + str(saved_run["at_gate"].get("llm_call_count"))
        + " calls where five were made."
    )
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    final_calls = saved_run["final_calls"]
    assert final_calls["status"] == 200 and isinstance(final_calls["body"], dict), (
        "Reading the run's model calls after the decision returned "
        + str(final_calls["status"])
        + " with "
        + str(final_calls["body"])[:300]
        + "."
    )
    calls = final_calls["body"]["calls"]
    called = [call.get("agent_name") for call in calls]
    added_agents = [config.AGENT_NAME_ORDER_WRITER, config.AGENT_NAME_UPDATE_WRITER]
    expected_count = config.LLM_CALLS_BEFORE_DECISION + len(added_agents)
    assert (called[config.LLM_CALLS_BEFORE_DECISION:] == added_agents
            and final_calls["body"]["call_count"] == len(calls) == expected_count
            and final_calls["body"]["expected_call_count"]
            == config.LLM_CALLS_BY_DECISION[config.DECISION_ORDER]), (
        "After the repair was ordered the run lists the calls "
        + str(called)
        + " ("
        + str(final_calls["body"].get("call_count"))
        + " of "
        + str(final_calls["body"].get("expected_call_count"))
        + " expected) where an order adds the works order and then the resident update to the "
        "five reviewing calls."
    )
    for call in calls:
        tokens = [call.get("prompt_tokens"), call.get("response_tokens"), call.get("total_tokens")]
        assert (all(isinstance(count, int) and count > 0 for count in tokens)
                and tokens[2] >= tokens[0] + tokens[1]
                and isinstance(call.get("latency_milliseconds"), int)
                and call["latency_milliseconds"] >= 0
                and isinstance(call.get("model_name"), str) and call["model_name"]), (
            "The call of "
            + str(call.get("agent_name"))
            + " is recorded as "
            + str({key: call.get(key) for key in ["model_name", "prompt_tokens", "response_tokens",
                                                  "total_tokens", "latency_milliseconds"]})
            + ". Every call is recorded with the model that answered, the tokens it was sent, "
            "the tokens it wrote and the total the model counted, and the milliseconds it "
            "took."
        )
    assert (final_calls["body"]["total_tokens"] == sum(call["total_tokens"] for call in calls)
            and final_calls["body"]["total_latency_milliseconds"]
            == sum(call["latency_milliseconds"] for call in calls)), (
        "The run's totals of "
        + str(final_calls["body"]["total_tokens"])
        + " tokens and "
        + str(final_calls["body"]["total_latency_milliseconds"])
        + " milliseconds are not the sums of its calls'."
    )
    with session_factory() as session:
        stored = session.query(LlmCall).filter(
            LlmCall.run_id == saved_run["at_gate"]["run_id"]).all()
    assert len(stored) == expected_count and sorted(
            row.agent_name for row in stored) == sorted(called), (
        "The llm_calls table holds "
        + str(len(stored))
        + " calls for the run, by "
        + str(sorted(row.agent_name for row in stored))
        + ", where the run made "
        + str(expected_count)
        + ", by "
        + str(sorted(called))
        + ". Each call is recorded once, under the run that made it."
    )


def test_review_keeps_tool_findings_and_open_recommendation(saved_run, saved_models,
                                                            defect_standards, live_files):
    from config import config
    from tools.prediction_tools import (describe_expected_crew_hours,
                                        estimate_expected_crew_hours,
                                        estimate_repeat_likelihood)
    from tools.report_data_tools import fetch_report
    from tools.screening_tools import count_live_street_works, screen_report

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    detail = saved_run["detail"]
    assert at_gate.get("decisions_open") == detail["screening"]["decisions_open"], (
        "The review offers the decisions "
        + str(at_gate.get("decisions_open"))
        + " where the screening of "
        + saved_run["report_reference"]
        + " leaves "
        + str(detail["screening"]["decisions_open"])
        + " open."
    )
    assert at_gate.get("recommended_decision") in at_gate.get("decisions_open", []), (
        "The review recommends "
        + repr(at_gate.get("recommended_decision"))
        + ", which is not among the decisions open "
        + str(at_gate.get("decisions_open"))
        + ". A recommendation the screening does not allow could never be taken."
    )
    write_ups = {key: str(at_gate.get(key) or "") for key in
                 ["review_plan_text", "requirements_analysis_text", "evidence_review_text",
                  "prediction_analysis_text", "recommendation_text"]}
    short = [key for key, text in write_ups.items() if len(text.split()) < 8]
    assert len(short) == 0, (
        "The review stores "
        + str(short)
        + " with fewer than eight words. Every reviewing agent's write-up is kept with the "
        "review for the Highways Inspector to read."
    )
    assert len(set(write_ups.values())) == len(write_ups), (
        "Two of the review's write-ups are the same text. Each agent writes its own part of "
        "the review."
    )
    assert (at_gate.get("coordinator_request") == saved_run["coordinator_request"]
            and at_gate.get("coordinator_id") == saved_run["opener"]["username"]), (
        "The review records the request "
        + repr(at_gate.get("coordinator_request"))
        + " opened by "
        + repr(at_gate.get("coordinator_id"))
        + " where "
        + saved_run["opener"]["username"]
        + " asked "
        + repr(saved_run["coordinator_request"])
        + "."
    )
    assert at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "The review was stored as "
        + repr(at_gate.get("status"))
        + " at the pause, where it waits as "
        + config.REVIEW_STATUS_AWAITING
        + "."
    )
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    record = at_gate
    screening = detail["screening"]
    register_count = detail["register_count"]
    report = fetch_report(live_files["reports"], saved_run["report_reference"])
    requirement_keys = ["defect_type", "standard_document", "requirement_kind",
                        "requirement_name", "requirement_sentence", "result", "blank_answers"]
    pairs = {
        "requirements": ([{key: requirement.get(key) for key in requirement_keys}
                          for requirement in record.get("requirements") or []],
                         [{key: requirement.get(key) for key in requirement_keys}
                          for requirement in screening["requirements"]]),
        "assessment": (record.get("assessment"), screening["assessment"]),
        "decisions_open": (record.get("decisions_open"), screening["decisions_open"]),
        "coordination_triggers": (record.get("coordination_triggers"),
                                  screening["coordination_triggers"]),
        "live_permits": (record.get("live_permits"), register_count["live_permits"]),
        "conflicting_permits": (record.get("conflicting_permits"),
                                register_count["conflicting_permits"]),
        "evidence": (record.get("evidence"), detail["evidence_findings"]),
        "disagreements": (record.get("disagreements"), detail["disagreements"]),
        "repeat_likelihood": (record.get("repeat_likelihood"),
                              detail["likelihood"]["repeat_likelihood"]),
        "likelihood_band": (record.get("likelihood_band"),
                            detail["likelihood"]["likelihood_band"]),
        "expected_crew_hours": (record.get("expected_crew_hours"),
                                detail["crew_hours"]["expected_crew_hours"]),
        "planned_crew_hours": (record.get("planned_crew_hours"),
                               detail["crew_hours"]["planned_crew_hours"]),
        "overrun_expected": (record.get("overrun_expected"),
                             detail["crew_hours"]["overrun_expected"]),
        "crew_hours_text": (record.get("crew_hours_text"),
                            describe_expected_crew_hours(detail["crew_hours"])),
        "defect_type": (record.get("defect_type"), report[config.DEFECT_TYPE_COLUMN]),
        "linked_defect_type": (record.get("linked_defect_type"),
                               report[config.LINKED_DEFECT_TYPE_COLUMN]),
        "street_name": (record.get("street_name"), report[config.STREET_NAME_COLUMN]),
        "district": (record.get("district"), report["district"]),
        "revision_number": (record.get("revision_number"), saved_run["revision_number"]),
    }
    for count_column in config.REQUIREMENT_COUNT_RESULTS:
        pairs[count_column] = (record.get(count_column), screening[count_column])
    differing = {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}
    assert len(differing) == 0, (
        "The stored review differs from the tools' findings for "
        + saved_run["report_reference"]
        + " in "
        + str({name: (str(pair[0])[:80], str(pair[1])[:80]) for name, pair in
               differing.items()})
        + ". The review records the screening under the report's own standards, the permits "
        "live on its street, the statement evidence, the flags, the likelihood and the "
        "expected crew hours the tools worked out, never the agents' retelling of them."
    )
    own_count = count_live_street_works(live_files["register"], report)
    for count_column in [config.LIVE_STREET_WORKS_COLUMN, config.CONFLICTING_STREET_WORKS_COLUMN]:
        report[count_column] = own_count[count_column]
    own_screening = screen_report(report, defect_standards["standards"])
    likelihood = estimate_repeat_likelihood(
        saved_models["repeat"], saved_models["repeat_schema"], report, own_screening)
    crew_hours = estimate_expected_crew_hours(
        saved_models["crew_hours"], saved_models["crew_hours_schema"], report, own_screening)
    assert (isinstance(own_screening, dict) and isinstance(likelihood, dict)
            and isinstance(crew_hours, dict)), (
        "Worked out again for "
        + saved_run["report_reference"]
        + ", the screening, the likelihood and the expected crew hours came back as "
        + type(own_screening).__name__
        + ", "
        + type(likelihood).__name__
        + " and "
        + type(crew_hours).__name__
        + ", where each is a record of its findings."
    )
    assert (likelihood.get("repeat_likelihood") == record.get("repeat_likelihood")
            and crew_hours.get("expected_crew_hours") == record.get("expected_crew_hours")
            and own_screening.get("assessment") == record.get("assessment")
            and own_screening.get("coordination_triggers") == record.get("coordination_triggers")
            and own_count.get("live_permits") == record.get("live_permits")), (
        "Worked out again from the live files, the saved models and the report's own "
        "standards, "
        + saved_run["report_reference"]
        + " is assessed "
        + str(own_screening.get("assessment"))
        + " with the triggers "
        + str(own_screening.get("coordination_triggers"))
        + ", the permits live "
        + str(own_count.get("live_permits"))
        + ", a likelihood of "
        + str(likelihood.get("repeat_likelihood"))
        + " and "
        + str(crew_hours.get("expected_crew_hours"))
        + " expected crew hours, where the review stored "
        + str(record.get("assessment"))
        + ", "
        + str(record.get("coordination_triggers"))
        + ", "
        + str(record.get("live_permits"))
        + ", "
        + str(record.get("repeat_likelihood"))
        + " and "
        + str(record.get("expected_crew_hours"))
        + "."
    )


def test_decorated_reply_read_whole():
    from agentic.language_model import parse_labelled_response

    labels = ["PRIORITY", "CHECKS", "RATIONALE"]
    decorated = ("Here is the plan you asked for.\n"
                 "**PRIORITY:** The depth requirement of both standards.\n"
                 "## Checks: 1. Confirm the depth against the photograph.\n"
                 "2. Confirm the location names a house number.\n"
                 "- RATIONALE: The request decides; the documents are evidence.\n"
                 "rationale: The answers are read as the Report Rules set out.")
    parsed = parse_labelled_response(decorated, labels, "review_planner_agent")
    assert (isinstance(parsed, dict) and sorted(parsed) == sorted(labels)
            and all(isinstance(value, str) for value in parsed.values())), (
        "Parsing a decorated reply returned "
        + str(parsed)[:300]
        + " where it returns exactly the fields "
        + str(labels)
        + ", each with its text."
    )
    assert (parsed["PRIORITY"] == "The depth requirement of both standards."
            and parsed["CHECKS"] == ("1. Confirm the depth against the photograph.\n"
                                     "2. Confirm the location names a house number.")
            and "The answers are read as the Report Rules set out." in parsed["RATIONALE"]
            and "The request decides" in parsed["RATIONALE"]), (
        "A reply with labels in bold or as headings, a field running over two lines and a "
        "label written twice was read as "
        + str(parsed)
        + ". Decoration around a label is not part of it, a line with no label belongs to the "
        "field above it, and a label written again adds to its field."
    )
    for reply, missing in [("PRIORITY: The depth.\nCHECKS: The photograph.", "RATIONALE"),
                           ("PRIORITY: The depth.\nCHECKS:\nRATIONALE: Because.", "CHECKS")]:
        refusal = None
        try:
            parse_labelled_response(reply, labels, "review_planner_agent")
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and missing in refusal and "review_planner_agent" in refusal, (
            "A reply that leaves "
            + missing
            + " out or empty gave "
            + str(refusal)
            + ". A reply missing a field is refused with a ValueError naming the agent and the "
            "field, never stored with a gap."
        )


def test_order_writes_works_order_and_resident_update(saved_run, live_files):
    from config import config
    from tools.report_data_tools import fetch_report
    from tools.review_document_tools import build_works_order_details

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert live_files["problem"] is None, live_files["problem"]
    final = saved_run["final_record"]
    assert final["status"] == 200 and len(final["body"]) == 1, (
        "The review whose repair was ordered could not be read back: "
        + str(final["body"])[:300]
    )
    record = final["body"][0]
    report = fetch_report(live_files["reports"], saved_run["report_reference"])
    built = build_works_order_details(report, saved_run["detail"]["screening"])
    details = record.get("document_details") or {}
    assert (isinstance(built, dict) and details == built
            and details.get(config.REPORT_REFERENCE_COLUMN) == saved_run["report_reference"]
            and details.get(config.DEFECT_TYPE_COLUMN) == report[config.DEFECT_TYPE_COLUMN]
            and details.get(config.STREET_NAME_COLUMN) == report[config.STREET_NAME_COLUMN]), (
        "The works order carries the details "
        + str(details)
        + " where build_works_order_details gives "
        + str(built)
        + " for the report and its screening. The details are added from the report and the "
        "screening, never written by the model, and name the report, its defect type, its "
        "street and where on it the defect is, with the measurements each standard asks for "
        "as the report gives them."
    )
    works_order = str(record.get("order_document_text") or "")
    resident_update = str(record.get("resident_update_text") or "")
    assert (len(works_order.split()) >= 8 and len(resident_update.split()) >= 8
            and works_order != resident_update), (
        "The order stored a works order of "
        + str(len(works_order.split()))
        + " words and a resident update of "
        + str(len(resident_update.split()))
        + ". An order writes the works order to the council's repair contractor and then a "
        "resident update for the person who reported the defect, two different texts."
    )
    assert record.get("coordination_brief") is None and record.get("status") == (
            config.REVIEW_STATUS_REPAIR_ORDERED), (
        "The review whose repair was ordered is stored as "
        + repr(record.get("status"))
        + " with a coordination brief of "
        + str(record.get("coordination_brief"))[:120]
        + ". An order is recorded as "
        + config.REVIEW_STATUS_REPAIR_ORDERED
        + " and sends nothing to the network coordination team."
    )


def test_workflow_paths_and_resume_from_checkpoint(saved_run, tmp_path):
    import os

    from agentic.state import ReportReviewState
    from config import config

    try:
        from agentic.graph import build_workflow

        compiled = build_workflow(str(tmp_path / "workflow_check.db"))
    except Exception as error:
        compiled = error
    builder = getattr(compiled, "builder", None)
    assert all(hasattr(builder, name) for name in ["nodes", "branches", "edges"]) and callable(
            getattr(compiled, "get_state", None)), (
        "Building the workflow gave "
        + repr(compiled)[:200]
        + ", where it returns the compiled workflow, built from its nodes, edges and "
        "conditional branches and kept in a checkpoint store."
    )
    node_names = sorted(str(name) for name in builder.nodes)
    expected_nodes = sorted(["load_report_context", "plan_report_review",
                             "assess_report_requirements", "analyse_report_statements",
                             "interpret_model_predictions", "prepare_report_recommendation",
                             "await_inspector_decision", "draft_order_document",
                             "draft_resident_update", "record_report_outcome"])
    assert node_names == expected_nodes, (
        "The workflow declares the nodes "
        + str(node_names)
        + " where the design has "
        + str(expected_nodes)
        + "."
    )
    destinations = {}
    for gate in ["await_inspector_decision", "draft_order_document"]:
        branches = builder.branches.get(gate, {})
        assert len(branches) == 1, (
            "The node "
            + gate
            + " has "
            + str(len(branches))
            + " conditional branches where it has one, choosing the path from the decision."
        )
        branch = list(branches.values())[0]
        decisions = config.DECISIONS
        if gate == "draft_order_document":
            decisions = [config.DECISION_ORDER, config.DECISION_INSPECT]
        for decision in decisions:
            state = {key: None for key in ReportReviewState.__annotations__}
            state["review_decision"] = {"decision": decision, "decided_by": "routing.check",
                                        "decided_by_role": config.ROLE_HIGHWAYS_INSPECTOR,
                                        "decided_by_district": config.TOWN_CENTRE,
                                        "decision_reason": "Routing is being checked."}
            routed = branch.path.invoke(state)
            if isinstance(routed, (list, tuple)) and len(routed) == 1:
                routed = routed[0]
            destinations[(gate, decision)] = branch.ends[routed] if branch.ends else routed
    expected_paths = {
        ("await_inspector_decision", config.DECISION_ORDER): "draft_order_document",
        ("await_inspector_decision", config.DECISION_INSPECT): "draft_order_document",
        ("await_inspector_decision", config.DECISION_REFER): "record_report_outcome",
        ("draft_order_document", config.DECISION_ORDER): "draft_resident_update",
        ("draft_order_document", config.DECISION_INSPECT): "record_report_outcome",
    }
    assert destinations == expected_paths, (
        "The decisions route to "
        + str(destinations)
        + ". An order writes the works order and then the resident update; an inspection "
        "writes the inspection request alone; a referral writes nothing and goes straight to "
        "the record, which adds the coordination brief."
    )
    edges = {(str(source), str(target)) for source, target in builder.edges}
    edges |= {(route[0], str(target)) for route, target in destinations.items()}
    reached = {"__start__"}
    grew = True
    while grew:
        grew = False
        for source, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                grew = True
    unreachable = sorted(set(node_names) - reached)
    assert len(unreachable) == 0, (
        "These nodes cannot be reached from the start: "
        + str(unreachable)
        + ". A node with no way in never runs, whatever it contains."
    )
    for source, target, reason in [
            ("__start__", "load_report_context",
             "every review starts from the tools' findings, before any agent runs"),
            ("prepare_report_recommendation", "await_inspector_decision",
             "the review is stored before the run waits for a Highways Inspector"),
            ("draft_resident_update", "record_report_outcome",
             "what was written has to reach the record"),
            ("record_report_outcome", "__end__",
             "the run ends once the outcome is recorded")]:
        assert (source, target) in edges, (
            "Nothing leads from " + source + " to " + target + ": " + reason + "."
        )
    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    stored_run = saved_run["run_record"]
    assert stored_run["problem"] is None, stored_run["problem"]
    expected_calls = config.LLM_CALLS_BY_DECISION[config.DECISION_ORDER]
    assert (stored_run["status"] == config.AGENT_RUN_STATUS_COMPLETE
            and stored_run["llm_call_count"] == expected_calls), (
        "After the repair was ordered the run is recorded as "
        + str(stored_run["status"])
        + " with "
        + str(stored_run["llm_call_count"])
        + " calls, where it closes "
        + config.AGENT_RUN_STATUS_COMPLETE
        + " with "
        + str(expected_calls)
        + "."
    )
    assert os.path.isfile(config.CHECKPOINT_DATABASE_FILE), (
        "There is no checkpoint at "
        + config.CHECKPOINT_DATABASE_FILE
        + ". A paused run waits on disk, so a decision taken later, or after the service "
        "restarts, finishes it."
    )
    try:
        snapshot = build_workflow(config.CHECKPOINT_DATABASE_FILE).get_state(
            {"configurable": {"thread_id": saved_run["at_gate"]["thread_id"]}})
    except Exception as error:
        snapshot = error
    assert isinstance(getattr(snapshot, "values", None), dict) and hasattr(snapshot, "next"), (
        "Reading the run back through a workflow built afresh on "
        + config.CHECKPOINT_DATABASE_FILE
        + " gave "
        + repr(snapshot)[:200]
        + ", where it gives the run's state and what it waits for."
    )
    values = snapshot.values
    decision = values.get("review_decision") or {}
    assert (len(snapshot.next) == 0 and decision.get("decision") == config.DECISION_ORDER
            and decision.get("decided_by") == saved_run["decider"]["username"]
            and decision.get("decided_by_district") == saved_run["decider"]["district"]
            and values.get("llm_call_count") == expected_calls), (
        "A workflow built afresh on the checkpoint file reads the run as waiting for "
        + str(list(snapshot.next))
        + " with the decision "
        + str(decision)
        + " and "
        + str(values.get("llm_call_count"))
        + " calls. The run was resumed from that file and finished there, with the Highways "
        "Inspector's decision and district kept in its state."
    )
    assert (values.get("review_id") == saved_run["at_gate"]["review_id"]
            and values.get("report_reference") == saved_run["report_reference"]
            and stored_run["thread_id"] == saved_run["at_gate"]["thread_id"]), (
        "The checkpoint's state belongs to "
        + str(values.get("review_id"))
        + " and "
        + str(values.get("report_reference"))
        + ", not to the review that was decided."
    )


def test_sign_in_and_out_decide_access(service_client):
    from access.password_security import hash_password
    from config import config
    from persistence.database import session_factory
    from persistence.models import UserAccount
    from persistence.repository import fetch_user_account

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    coordinators = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_STREET_CARE_COORDINATOR]
    account = coordinators[1]
    other_account = coordinators[2]
    credentials = {"username": account["username"], "password": account["password"]}
    signed_in = client.post("/auth/signin", json=credentials)
    body = signed_in.json() if signed_in.status_code == 200 else {}
    assert ({"username", "display_name", "role", "district", "session_token"} <= set(body)
            and len(str(body["session_token"]).strip()) > 0), (
        "Signing in as "
        + account["username"]
        + " with the password the configuration seeds returned "
        + str(signed_in.status_code)
        + " with "
        + signed_in.text[:200]
        + ", where a sign-in answers with the user's name, display name, role, district and a "
        "session token."
    )
    assert (body["username"] == account["username"] and body["role"] == account["role"]
            and body["display_name"] == account["display_name"]
            and body["district"] == account["district"]), (
        "Signing in as "
        + account["username"]
        + " answered with the name "
        + repr(body["username"])
        + ", the display name "
        + repr(body["display_name"])
        + ", the role "
        + repr(body["role"])
        + " and the district "
        + repr(body["district"])
        + ", which are not those of the account the configuration seeds."
    )
    wrong = client.post("/auth/signin", json={"username": account["username"],
                                              "password": "wrong-" + account["password"]})
    stranger = client.post("/auth/signin", json={"username": "no-such-person",
                                                 "password": account["password"]})
    assert (wrong.status_code == 401 and stranger.status_code == 401
            and wrong.json() == stranger.json() == {"detail": config.SIGN_IN_FAILED_MESSAGE}), (
        "A wrong password was answered "
        + str(wrong.status_code)
        + " with "
        + wrong.text[:200]
        + " and an unknown username "
        + str(stranger.status_code)
        + " with "
        + stranger.text[:200]
        + ". Both are refused with 401 and the same words, "
        + repr(config.SIGN_IN_FAILED_MESSAGE)
        + ", so a stranger cannot tell which usernames exist."
    )
    parameters = {"defect_type": config.QUEUE_DEFECT_TYPE_ALL}
    first_headers = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    accepted = client.get("/queue", params=parameters, headers=first_headers)
    assert accepted.status_code == 200, (
        "The token just given to "
        + account["username"]
        + " was answered "
        + str(accepted.status_code)
        + " by the queue. A fresh session token opens what its role is permitted."
    )
    for meaning, headers in [("with no token", None),
                             ("with a token nobody was given",
                              {config.SESSION_TOKEN_HEADER: "not-a-token-anyone-was-given"})]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert (refused.status_code == 401
                and refused.json() == {"detail": config.SIGN_IN_REQUIRED_MESSAGE}), (
            "The queue answered a request "
            + meaning
            + " with "
            + str(refused.status_code)
            + " and "
            + refused.text[:200]
            + ". Only a signed-in user is served; anyone else is refused with 401 and "
            + repr(config.SIGN_IN_REQUIRED_MESSAGE)
            + "."
        )
    again = client.post("/auth/signin", json=credentials)
    second_token = again.json().get("session_token") if again.status_code == 200 else None
    second_headers = {config.SESSION_TOKEN_HEADER: str(second_token)}
    replaced = client.get("/queue", params=parameters, headers=first_headers)
    current = client.get("/queue", params=parameters, headers=second_headers)
    assert (again.status_code == 200 and replaced.status_code == 401
            and current.status_code == 200), (
        "After "
        + account["username"]
        + " signed in a second time, answered "
        + str(again.status_code)
        + ", the first token got "
        + str(replaced.status_code)
        + " from the queue and the second "
        + str(current.status_code)
        + ". An account holds one session: signing in somewhere else ends the session before "
        "it."
    )
    signed_out = client.post("/auth/signout", headers=second_headers)
    after = client.get("/queue", params=parameters, headers=second_headers)
    twice = client.post("/auth/signout", headers=second_headers)
    assert (signed_out.status_code == 200
            and signed_out.json() == {"username": account["username"], "signed_out": True}
            and after.status_code == 401 and twice.status_code == 401), (
        "Signing out answered "
        + str(signed_out.status_code)
        + " with "
        + signed_out.text[:200]
        + "; the token then got "
        + str(after.status_code)
        + " from the queue and "
        + str(twice.status_code)
        + " when it signed out again. Signing out answers with the username and signed_out "
        "true, and the token it ends opens nothing afterwards."
    )
    with session_factory() as session:
        stored = fetch_user_account(session, UserAccount.username == account["username"])
        other = fetch_user_account(session, UserAccount.username == other_account["username"])
    missing = sorted({"password_hash", "password_salt"}
                     - (set(stored if isinstance(stored, dict) else {})
                        & set(other if isinstance(other, dict) else {})))
    assert len(missing) == 0, (
        "fetch_user_account read "
        + account["username"]
        + " and "
        + other_account["username"]
        + " back without "
        + str(missing)
        + ", so how their passwords are kept cannot be read."
    )
    assert (stored["password_hash"] != account["password"]
            and stored["password_hash"] == hash_password(account["password"],
                                                         stored["password_salt"])), (
        "The password stored for "
        + account["username"]
        + " is not the salted hash of the password. A password kept as written is read by "
        "anyone who can open the database."
    )
    assert (stored["password_salt"] != other["password_salt"]
            and stored["password_hash"] != other["password_hash"]), (
        account["username"]
        + " and "
        + other_account["username"]
        + " share a password and are stored with the same salt or the same hash, so one "
        "cracked hash opens both accounts."
    )


def test_roles_reach_permissions_and_districts(service_client, staff, live_files):
    from fastapi import HTTPException

    from access.permissions import authorize_district
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun

    assert service_client["problem"] is None, service_client["problem"]
    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    client = service_client["client"]
    signed_in = {}
    for role_name, signed in [
            (config.ROLE_HIGHWAYS_INSPECTOR,
             staff["inspectors"].get(config.TOWN_CENTRE, [{}, {}])[1]),
            (config.ROLE_STREET_CARE_COORDINATOR, staff["coordinator"]),
            (config.ROLE_CUSTOMER_SERVICES_ADVISER, staff["adviser"])]:
        assert {"username", "role", "district", "pages", "permissions", "headers"} <= set(
                signed), (
            "A signed-in "
            + role_name
            + " came back as "
            + str({key: value for key, value in signed.items() if key != "headers"})[:200]
            + ", where a sign-in answers with the username, role, district and the pages and "
            "permissions of the user's role."
        )
        assert (signed["role"] == role_name
                and sorted(signed["pages"]) == sorted(config.ROLE_PAGE_ACCESS[role_name])
                and sorted(signed["permissions"]) == sorted(config.ROLE_PERMISSIONS[role_name])), (
            "Signing in as a "
            + role_name
            + " answered with the role "
            + str(signed["role"])
            + ", the pages "
            + str(signed["pages"])
            + " and the permissions "
            + str(signed["permissions"])
            + ", where the configuration grants that role the pages "
            + str(config.ROLE_PAGE_ACCESS[role_name])
            + " and the permissions "
            + str(config.ROLE_PERMISSIONS[role_name])
            + "."
        )
        signed_in[role_name] = (signed, signed["headers"])
    unknown_review = config.REVIEW_ID_PREFIX + "DOES-NOT-EXIST"
    probes = [
        (config.PERMISSION_QUEUE, "GET", "/queue",
         {"defect_type": config.QUEUE_DEFECT_TYPE_ALL}, None, 200),
        (config.PERMISSION_QUEUE, "GET", "/queue/SC0000000", None, None, 404),
        (config.PERMISSION_REVIEW, "POST", "/reviews", None,
         {"report_reference": "SC0000000",
          "coordinator_request": "Are the requirements met on the report?"}, 404),
        (config.PERMISSION_DECISION, "POST", "/reviews/" + unknown_review + "/decision", None,
         {"decision": config.DECISION_REFER,
          "decision_reason": "A coordination trigger applies to this report."}, 404),
        (config.PERMISSION_AUDIT, "GET", "/reviews",
         {"status_value": [config.REVIEW_STATUS_AWAITING]}, None, 200),
        (config.PERMISSION_AUDIT, "GET", "/reviews/" + unknown_review + "/calls", None, None,
         404),
        (config.PERMISSION_AUDIT, "GET", "/audit/chat-turns", {"turn_limit": 1}, None, 200),
        (config.PERMISSION_CHAT, "GET", "/chat/turns", {"turn_limit": 1}, None, 200),
    ]
    for role_name, (_account, role_headers) in signed_in.items():
        for permission, method, path, parameters, payload, status_when_granted in probes:
            reply = client.request(method, path, params=parameters, json=payload,
                                   headers=role_headers)
            if permission in config.ROLE_PERMISSIONS[role_name]:
                assert reply.status_code == status_when_granted, (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + " where it gets "
                    + str(status_when_granted)
                    + ". The "
                    + permission
                    + " permission is granted to that role, so the guard lets it through and "
                    "the endpoint answers: a report or a review that does not exist is answered "
                    "404, before any language model call."
                )
            else:
                expected_refusal = config.PERMISSION_REFUSED_MESSAGE.format(
                    role=role_name, permission=permission)
                assert (reply.status_code == 403
                        and reply.json().get("detail") == expected_refusal), (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + ". The "
                    + permission
                    + " permission is not granted to that role, so the guard refuses it with 403 "
                    "and the configuration's words, "
                    + repr(expected_refusal)
                    + ", whether or not the thing asked for exists."
                )
    inspector, inspector_headers = signed_in[config.ROLE_HIGHWAYS_INSPECTOR]
    own_streets = config.INSPECTION_DISTRICT_STREETS[inspector["district"]]
    reports = live_files["reports"]
    outside = reports[~reports[config.STREET_NAME_COLUMN].isin(own_streets)].head(1)
    inside = reports[reports[config.STREET_NAME_COLUMN].isin(own_streets)].head(1)
    assert len(outside) == 1 and len(inside) == 1, (
        "The live queue holds no report inside, or none outside, the streets of "
        + inspector["district"]
        + ", where it holds both."
    )
    outside_reference = outside[config.REPORT_REFERENCE_COLUMN].iloc[0]
    outside_street = outside[config.STREET_NAME_COLUMN].iloc[0]
    with session_factory() as session:
        runs_before = session.query(AgentRun).filter(
            AgentRun.report_reference == outside_reference,
            AgentRun.coordinator_id == inspector["username"]).count()
    refused = client.post("/reviews", json={"report_reference": outside_reference,
                                            "coordinator_request": "Is every requirement met?"},
                          headers=inspector_headers)
    with session_factory() as session:
        runs_after = session.query(AgentRun).filter(
            AgentRun.report_reference == outside_reference,
            AgentRun.coordinator_id == inspector["username"]).count()
    expected_refusal = config.DISTRICT_REFUSED_MESSAGE.format(
        username=inspector["username"], district=inspector["district"],
        street_name=outside_street, report_reference=outside_reference)
    assert (refused.status_code == 403 and refused.json().get("detail") == expected_refusal
            and runs_after == runs_before), (
        "A Highways Inspector appointed for "
        + inspector["district"]
        + " opening a review of "
        + outside_reference
        + ", on "
        + outside_street
        + ", was answered "
        + str(refused.status_code)
        + " with "
        + refused.text[:200]
        + ", and the runs they opened on it went from "
        + str(runs_before)
        + " to "
        + str(runs_after)
        + ". A Highways Inspector acts only on the streets of their own district, so it is "
        "refused with 403 and "
        + repr(expected_refusal)
        + " before any run starts."
    )
    coordinator_account = signed_in[config.ROLE_STREET_CARE_COORDINATOR][0]
    outcomes = {}
    for meaning, user, street_name, reference in [
            ("a Highways Inspector inside their district", inspector,
             inside[config.STREET_NAME_COLUMN].iloc[0],
             inside[config.REPORT_REFERENCE_COLUMN].iloc[0]),
            ("a Highways Inspector outside their district", inspector, outside_street,
             outside_reference),
            ("a Street Care Coordinator anywhere", coordinator_account, outside_street,
             outside_reference)]:
        try:
            outcomes[meaning] = authorize_district(user, street_name, reference)
        except HTTPException as error:
            outcomes[meaning] = (error.status_code, error.detail)
        except Exception as error:
            outcomes[meaning] = type(error).__name__ + ": " + str(error)[:120]
    expected_outcomes = {"a Highways Inspector inside their district": None,
                         "a Highways Inspector outside their district": (403, expected_refusal),
                         "a Street Care Coordinator anywhere": None}
    assert outcomes == expected_outcomes, (
        "authorize_district answered "
        + str(outcomes)
        + " where it answers "
        + str(expected_outcomes)
        + ". It lets a Highways Inspector act on the streets of their district and refuses them "
        "with 403 on any other street, and a Street Care Coordinator, who belongs to no "
        "district, is not held to one."
    )


def test_queue_order_and_rows_match_detail(service_client, staff, live_files, saved_models,
                                           defect_standards, language_model_requests):
    from config import config
    from tools.prediction_tools import estimate_expected_crew_hours, estimate_repeat_likelihood
    from tools.report_data_tools import fetch_report, rank_report_queue
    from tools.screening_tools import count_live_street_works, screen_report

    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert defect_standards["problem"] is None, defect_standards["problem"]
    footway, unlinked = config.FOOTWAY, config.NO_LINKED_DEFECT
    planted = live_files["reports"].head(5).copy()
    planted[config.REPORT_REFERENCE_COLUMN] = ["SC0500004", "SC0500002", "SC0500001",
                                               "SC0500003", "SC0500000"]
    planted[config.DEFECT_TYPE_COLUMN] = [config.CARRIAGEWAY, footway, config.DRAINAGE,
                                          footway, config.VEGETATION]
    planted[config.LINKED_DEFECT_TYPE_COLUMN] = [unlinked, unlinked, footway,
                                                 config.STREET_LIGHTING, unlinked]
    ranked = rank_report_queue(planted)
    ranked = list(ranked) if isinstance(ranked, (list, tuple)) else ranked
    assert ranked == ["SC0500001", "SC0500002", "SC0500003", "SC0500000", "SC0500004"], (
        "rank_report_queue put planted reports - a carriageway defect SC0500004, a footway "
        "defect SC0500002, a drainage defect SC0500001 linked to the footway, a footway defect "
        "SC0500003 linked to a street light and a vegetation defect SC0500000 - in the order "
        + str(ranked)
        + ". Reports whose defect type or linked type is the footway come first, the rest after "
        "them, and each group runs by report reference."
    )
    client = service_client["client"]
    headers = staff["coordinator"]["headers"]
    reply = client.get("/queue", params={"defect_type": config.QUEUE_DEFECT_TYPE_ALL},
                       headers=headers)
    assert reply.status_code == 200 and isinstance(reply.json(), list), (
        "The queue returned "
        + str(reply.status_code)
        + " with "
        + reply.text[:200]
        + " to a signed-in Street Care Coordinator."
    )
    everything = reply.json()
    missing = sorted({"queue_position", "report_reference", "revision_number", "defect_type",
                      "linked_defect_type", "street_name", "district", "report_channel",
                      "report_status", "document_count", "requirements_met_count",
                      "requirements_not_met_count", "requirements_not_documented_count",
                      "live_permit_count", "conflicting_permit_count",
                      "coordination_trigger_count", "assessment", "repeat_likelihood",
                      "likelihood_band", "expected_crew_hours", "overrun_expected",
                      "disagreement_count"}
                     - set(everything[0] if everything else {}))
    assert len(missing) == 0, (
        "A queue row carries no "
        + str(missing)
        + ", so the queue cannot be ordered, filtered or read."
    )
    references = [row["report_reference"] for row in everything]
    reports = live_files["reports"]
    open_references = list(reports[config.REPORT_REFERENCE_COLUMN])
    assert len(set(references)) == len(references) and sorted(references) == sorted(
            open_references), (
        "The queue holds "
        + str(len(references))
        + " rows for "
        + str(len(set(references)))
        + " reports, where the open defect reports, read through the cleaning, number "
        + str(len(open_references))
        + ". Every open report is in the queue exactly once, at its latest revision."
    )
    footway_reports = set(reports.loc[(reports[config.DEFECT_TYPE_COLUMN] == footway)
                                      | (reports[config.LINKED_DEFECT_TYPE_COLUMN] == footway),
                                      config.REPORT_REFERENCE_COLUMN])
    flags = [row["report_reference"] in footway_reports for row in everything]
    first_other = flags.index(False) if False in flags else len(flags)
    first_block, other_block = references[:first_other], references[first_other:]
    assert (not any(flags[first_other:]) and len(footway_reports) > 0
            and set(first_block) == footway_reports
            and first_block == sorted(first_block)
            and other_block == sorted(other_block)), (
        "The queue opens with "
        + str(first_block[:6])
        + " and then runs "
        + str(other_block[:6])
        + " ..., where the open reports of a footway defect, as the defect type or the linked "
        "type, are "
        + str(sorted(footway_reports))
        + ". They come first and then the rest, each group in the order of its report "
        "references, so the Street Care Coordinators work the queue from the top."
    )
    assert [row["queue_position"] for row in everything] == list(range(1, len(everything) + 1)), (
        "The queue positions run "
        + str([row["queue_position"] for row in everything][:10])
        + " ... where they count from 1 down the queue."
    )
    in_review = language_model_requests["review"].get("report_reference")
    checked_against_tools = [references[0], references[len(references) // 2], references[-1]]
    requirement_keys = ["defect_type", "standard_document", "requirement_kind",
                        "requirement_name", "requirement_sentence", "result", "blank_answers"]
    differing = {}
    for row in everything:
        detail_reply = client.get("/queue/" + row["report_reference"], headers=headers)
        detail = detail_reply.json() if detail_reply.status_code == 200 else {}
        described = detail.get("report", {})
        shown = detail.get("screening", {})
        shown_count = detail.get("register_count", {})
        shown_likelihood = detail.get("likelihood", {})
        shown_hours = detail.get("crew_hours", {})
        expected = {
            "queue_position": detail.get("queue_position"),
            "report_status": detail.get("report_status"),
            "defect_type": described.get(config.DEFECT_TYPE_COLUMN),
            "linked_defect_type": described.get(config.LINKED_DEFECT_TYPE_COLUMN),
            "street_name": described.get(config.STREET_NAME_COLUMN),
            "district": described.get("district"),
            "report_channel": described.get("report_channel"),
            "document_count": len(detail.get("report_documents", [])),
            "requirements_met_count": shown.get(config.REQUIREMENTS_MET_COUNT_COLUMN),
            "requirements_not_met_count": shown.get(config.REQUIREMENTS_NOT_MET_COUNT_COLUMN),
            "requirements_not_documented_count": shown.get(
                config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN),
            "live_permit_count": shown_count.get(config.LIVE_STREET_WORKS_COLUMN),
            "conflicting_permit_count": shown_count.get(config.CONFLICTING_STREET_WORKS_COLUMN),
            "coordination_trigger_count": len(shown.get("coordination_triggers", [])),
            "assessment": shown.get("assessment"),
            "repeat_likelihood": shown_likelihood.get("repeat_likelihood"),
            "likelihood_band": shown_likelihood.get("likelihood_band"),
            "expected_crew_hours": shown_hours.get("expected_crew_hours"),
            "overrun_expected": shown_hours.get("overrun_expected"),
            "disagreement_count": len(detail.get("disagreements", [])),
        }
        for key, value in expected.items():
            if key == "report_status" and row["report_reference"] == in_review:
                continue
            if row[key] != value:
                differing[row["report_reference"]] = (key + " is " + str(row[key])[:60]
                                                      + " where its detail gives "
                                                      + str(value)[:60])
        if str(row["revision_number"]) != described.get(config.REVISION_NUMBER_COLUMN):
            differing[row["report_reference"]] = ("revision_number is "
                                                  + str(row["revision_number"])
                                                  + " where its detail gives "
                                                  + str(described.get(
                                                      config.REVISION_NUMBER_COLUMN)))
        if row["report_reference"] in checked_against_tools:
            report = fetch_report(reports, row["report_reference"])
            register_count = count_live_street_works(live_files["register"], report)
            for count_column in [config.LIVE_STREET_WORKS_COLUMN,
                                 config.CONFLICTING_STREET_WORKS_COLUMN]:
                report[count_column] = register_count[count_column]
            screening = screen_report(report, defect_standards["standards"])
            assert isinstance(screening, dict) and isinstance(
                    screening.get("requirements"), list), (
                "screen_report returned "
                + str(screening)[:200]
                + " for "
                + row["report_reference"]
                + ", where it returns the report's screening."
            )
            for key in (["report_reference", "defect_types", "coordination_triggers",
                         "assessment", "decisions_open"]
                        + list(config.REQUIREMENT_COUNT_RESULTS)):
                if shown.get(key) != screening.get(key):
                    differing[row["report_reference"]] = ("the detail's " + key
                                                          + " is not the tools' own")
            for key in ["live_permits", "conflicting_permits"]:
                if shown_count.get(key) != register_count.get(key):
                    differing[row["report_reference"]] = ("the detail's " + key
                                                          + " is not the register's count")
            own_requirements = [{key: requirement.get(key) for key in requirement_keys}
                                for requirement in screening["requirements"]
                                if isinstance(requirement, dict)]
            shown_requirements = [{key: requirement.get(key) for key in requirement_keys}
                                  for requirement in shown.get("requirements", [])
                                  if isinstance(requirement, dict)]
            if shown_requirements != own_requirements:
                differing[row["report_reference"]] = ("the detail's requirements are not "
                                                      "the tools'")
            likelihood = estimate_repeat_likelihood(
                saved_models["repeat"], saved_models["repeat_schema"], report, screening)
            crew_hours = estimate_expected_crew_hours(
                saved_models["crew_hours"], saved_models["crew_hours_schema"], report, screening)
            if shown_likelihood != likelihood or shown_hours != crew_hours:
                differing[row["report_reference"]] = ("the detail's figures are not the saved "
                                                      "models'")
    assert len(differing) == 0, (
        str(len(differing))
        + " rows of the queue differ from their detail or from the tools, for example "
        + str(dict(list(differing.items())[:3]))
        + ". Each row shows what its detail shows: its report's own screening under its "
        "standards, the permits live on its street in the register, the figures the saved "
        "models give it, the documents attached to it and the statement sentences flagged in "
        "them."
    )
    for defect_type in config.QUEUE_DEFECT_TYPE_OPTIONS[1:]:
        filtered_reply = client.get("/queue", params={"defect_type": defect_type},
                                    headers=headers)
        filtered = filtered_reply.json() if filtered_reply.status_code == 200 else []
        wanted_rows = [row for row in everything
                       if defect_type in (row["defect_type"], row["linked_defect_type"])]
        filtered_settled = [{key: value for key, value in row.items()
                             if key != "report_status" or row["report_reference"] != in_review}
                            for row in filtered]
        wanted_settled = [{key: value for key, value in row.items()
                           if key != "report_status" or row["report_reference"] != in_review}
                          for row in wanted_rows]
        assert len(filtered) > 0 and filtered_settled == wanted_settled, (
            "Filtering the queue to "
            + defect_type
            + " did not return exactly the reports of the full queue whose defect type or "
            "linked type is "
            + defect_type
            + ", in the same order and with the same positions."
        )
    for meaning, parameters in [("a defect type Wrenford has no standard for",
                                 {"defect_type": "Graffiti"}),
                                ("no defect type at all", None)]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert refused.status_code == 422, (
            "Asking for the queue with "
            + meaning
            + " returned "
            + str(refused.status_code)
            + " rather than 422. The filter takes a defect type the configuration names, or all "
            "of them, and nothing else."
        )
    unknown = client.get("/queue/SC0000000", headers=headers)
    expected_unknown = config.UNKNOWN_REPORT_MESSAGE.format(report_reference="SC0000000")
    assert unknown.status_code == 404 and unknown.json().get("detail") == expected_unknown, (
        "The detail of SC0000000, which is not an open report, returned "
        + str(unknown.status_code)
        + " with "
        + unknown.text[:200]
        + " where it is 404 with "
        + repr(expected_unknown)
        + "."
    )


def test_run_waits_for_appointed_inspector(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    undecided = ["decision", "decided_by", "decided_by_role", "decided_by_district",
                 "decision_reason", "order_document_text", "document_details",
                 "resident_update_text", "coordination_brief"]
    written = [field for field in undecided if at_gate.get(field) is not None]
    assert len(written) == 0 and at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "At the pause the review already holds "
        + str(written)
        + " with the status "
        + repr(at_gate.get("status"))
        + ". Nothing is decided, ordered or written until a Highways Inspector decides."
    )
    opener, outsider = saved_run["opener"], saved_run["outsider"]
    street_name = saved_run["detail"]["report"][config.STREET_NAME_COLUMN]
    for refused_key, expected_status, expected_refusal, meaning in [
            ("own_decision", 409, config.OWN_REVIEW_MESSAGE.format(
                decided_by=opener["username"], report_reference=saved_run["report_reference"]),
             "The Highways Inspector who opened the review, deciding it,"),
            ("outside_decision", 403, config.DISTRICT_REFUSED_MESSAGE.format(
                username=outsider["username"], district=outsider["district"],
                street_name=street_name, report_reference=saved_run["report_reference"]),
             "A Highways Inspector appointed for " + str(outsider["district"])
             + ", deciding a report on " + str(street_name) + ",")]:
        refused = saved_run[refused_key]
        assert (refused["status"] == expected_status and isinstance(refused["body"], dict)
                and refused["body"].get("detail") == expected_refusal), (
            meaning
            + " was answered "
            + str(refused["status"])
            + " with "
            + str(refused["body"])[:300]
            + ", where the service refuses with "
            + str(expected_status)
            + " and the configuration's words: "
            + repr(expected_refusal)
            + ". The Highways Policy has a Highways Inspector appointed for the street's "
            "district, who did not open the review, decide it."
        )
        assert (refused["calls_before"] == refused["calls_after"]
                == config.LLM_CALLS_BEFORE_DECISION), (
            "The refused decision moved the call count from "
            + str(refused["calls_before"])
            + " to "
            + str(refused["calls_after"])
            + ". A decision is checked before the run resumes, so a refused one spends no call."
        )


def test_decision_once_refusals_before_calls(saved_run, service_client):
    from config import config
    from persistence.database import session_factory
    from persistence.models import ReviewDecision
    from tools.decision_tools import find_decision_refusal, find_review_conflict

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    for name, meaning in [("short_reason", "a reason of five characters"),
                          ("unknown_decision", "a decision that is none of the three"),
                          ("short_request", "a request of two characters")]:
        assert saved_run[name]["status"] == 422, (
            "Sending "
            + meaning
            + " was answered "
            + str(saved_run[name]["status"])
            + " with "
            + str(saved_run[name]["body"])[:200]
            + ". The request's schema refuses it with 422 before anything is looked up."
        )
    second = saved_run["second_review"]
    expected_second = config.REPORT_ALREADY_REVIEWED_MESSAGE.format(
        revision_number=saved_run["revision_number"],
        report_reference=saved_run["report_reference"])
    assert (second["status"] == 409 and second["body"].get("detail") == expected_second
            and saved_run["run_record"].get("runs_opened") == 1), (
        "Opening a second review of "
        + saved_run["report_reference"]
        + " was answered "
        + str(second["status"])
        + " with "
        + str(second["body"])[:200]
        + ", and this run of the suite recorded "
        + str(saved_run["run_record"].get("runs_opened"))
        + " runs of the report. A revision of a report in the queue is reviewed once, and a "
        "second review is refused before a run starts: "
        + repr(expected_second)
    )
    decided = saved_run["decided"]["body"]
    decider = saved_run["decider"]
    assert (decided.get("decision") == config.DECISION_ORDER
            and decided.get("decided_by") == decider["username"]
            and decided.get("decided_by_role") == config.ROLE_HIGHWAYS_INSPECTOR
            and decided.get("decided_by_district") == decider["district"]
            and decided.get("decision_reason") == saved_run["decision_reason"]
            and decided.get("status") == config.REVIEW_STATUS_REPAIR_ORDERED), (
        "The order came back recorded as "
        + str({key: decided.get(key) for key in ["decision", "decided_by", "decided_by_role",
                                                 "decided_by_district", "decision_reason",
                                                 "status"]})
        + ", which is not the decision the Highways Inspector sent."
    )
    review_id = saved_run["at_gate"]["review_id"]
    again = service_client["client"].post(
        "/reviews/" + review_id + "/decision",
        json={"decision": config.DECISION_ORDER, "decision_reason": saved_run["decision_reason"]},
        headers=decider["headers"])
    expected_again = config.ALREADY_DECIDED_MESSAGE.format(
        report_reference=saved_run["report_reference"])
    assert again.status_code == 409 and again.json().get("detail") == expected_again, (
        "Deciding the review a second time was answered "
        + str(again.status_code)
        + " with "
        + again.text[:200]
        + ". A review is decided once: "
        + repr(expected_again)
    )
    with session_factory() as session:
        stored = session.query(ReviewDecision).filter(ReviewDecision.review_id == review_id).all()
    assert len(stored) == 1 and stored[0].decision == config.DECISION_ORDER, (
        "The review_decisions table holds "
        + str(len(stored))
        + " decisions for the review where it holds the one decision taken."
    )
    reference = "SC0950107"
    report = {config.REPORT_REFERENCE_COLUMN: reference,
              config.STREET_NAME_COLUMN: config.INSPECTION_DISTRICT_STREETS[config.TOWN_CENTRE][0]}
    street_name = report[config.STREET_NAME_COLUMN]
    sound = {"decision": config.DECISION_ORDER, "decided_by": "owen",
             "decided_by_role": config.ROLE_HIGHWAYS_INSPECTOR,
             "decided_by_district": config.TOWN_CENTRE,
             "decision_reason": "Every requirement is met on the report."}
    ordered_screening = {"report_reference": reference, "assessment": config.DECISION_ORDER,
                         "decisions_open": config.DECISIONS_OPEN[config.DECISION_ORDER]}
    inspect_screening = dict(ordered_screening, assessment=config.DECISION_INSPECT,
                             decisions_open=config.DECISIONS_OPEN[config.DECISION_INSPECT])
    refer_screening = dict(ordered_screening, assessment=config.DECISION_REFER,
                           decisions_open=config.DECISIONS_OPEN[config.DECISION_REFER])
    cases = [
        ("an order of a report assessed INSPECT ON SITE", sound, inspect_screening,
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_ORDER, report_reference=reference,
             reason=config.ORDER_CLOSED_REASONS[config.DECISION_INSPECT])),
        ("an order of a report assessed REFER FOR COORDINATION", sound, refer_screening,
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_ORDER, report_reference=reference,
             reason=config.ORDER_CLOSED_REASONS[config.DECISION_REFER])),
        ("a decision with no reason", dict(sound, decision_reason="  "), ordered_screening,
         config.DECISION_FIELDS_MISSING_MESSAGE.format(report_reference=reference,
                                                       fields="decision_reason")),
        ("a decision that is none of the three", dict(sound, decision="PATCH IT"),
         ordered_screening, config.UNKNOWN_DECISION_MESSAGE.format(
             decision="PATCH IT", report_reference=reference,
             decisions=", ".join(config.DECISIONS))),
        ("a decision by a Street Care Coordinator",
         dict(sound, decided_by_role=config.ROLE_STREET_CARE_COORDINATOR), ordered_screening,
         config.DECIDING_ROLE_MESSAGE.format(
             role=config.ROLE_STREET_CARE_COORDINATOR, report_reference=reference,
             inspector_role=config.ROLE_HIGHWAYS_INSPECTOR)),
        ("a decision by a Highways Inspector appointed for the North district",
         dict(sound, decided_by="priya", decided_by_district=config.NORTH),
         ordered_screening, config.NOT_APPOINTED_MESSAGE.format(
             decided_by="priya", district=config.NORTH, street_name=street_name,
             report_reference=reference)),
        ("an inspection of a report assessed ORDER REPAIR",
         dict(sound, decision=config.DECISION_INSPECT), ordered_screening, None),
        ("a referral of a report assessed INSPECT ON SITE",
         dict(sound, decision=config.DECISION_REFER), inspect_screening, None),
        ("an inspection of a report assessed REFER FOR COORDINATION",
         dict(sound, decision=config.DECISION_INSPECT), refer_screening, None),
        ("a sound order", sound, ordered_screening, None),
    ]
    for meaning, review_decision, screening, expected_refusal in cases:
        refusal = find_decision_refusal(review_decision, report, screening, "amelia")
        assert refusal == expected_refusal, (
            "find_decision_refusal answered "
            + meaning
            + " with "
            + repr(refusal)
            + " where it answers "
            + repr(expected_refusal)
            + ". Every decision the workflow's gate would refuse is named before the run "
            "resumes, in the configuration's words, so no call is spent on it."
        )
    other = dict(sound, decision=config.DECISION_INSPECT)
    for meaning, review_status, pending, expected_conflict in [
            ("a decision on a review whose repair was ordered",
             config.REVIEW_STATUS_REPAIR_ORDERED, None,
             config.ALREADY_DECIDED_MESSAGE.format(report_reference=reference)),
            ("a decision on a review already referred", config.REVIEW_STATUS_REFERRED, None,
             config.ALREADY_DECIDED_MESSAGE.format(report_reference=reference)),
            ("an order while owen's inspection is still being recorded",
             config.REVIEW_STATUS_AWAITING, other,
             config.DECISION_IN_PROGRESS_MESSAGE.format(
                 pending_decided_by="owen", pending_decision=config.DECISION_INSPECT,
                 report_reference=reference)),
            ("the same decision sent again to finish it", config.REVIEW_STATUS_AWAITING, sound,
             None),
            ("a decision on a review awaiting one", config.REVIEW_STATUS_AWAITING, None, None)]:
        conflict = find_review_conflict(review_status, pending, sound, reference)
        assert conflict == expected_conflict, (
            "find_review_conflict answered "
            + meaning
            + " with "
            + repr(conflict)
            + " where it answers "
            + repr(expected_conflict)
            + ". A review is decided once, and a decision still being recorded is finished only "
            "by the same decision sent again."
        )


def test_chat_answers_from_documents_and_cites_them(chat_replies, knowledge_collection,
                                                    service_client, staff):
    import os

    from config import config
    from knowledge.vector_index import search_documents

    assert chat_replies["problem"] is None, chat_replies["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    for name, meaning in [("standard", "Asking the street care assistant what the footway "
                                       "defect standard asks for"),
                          ("turns", "Reading back the Street Care Coordinator's own chat turns")]:
        assert chat_replies[name]["status"] == 200, (
            meaning
            + " returned "
            + str(chat_replies[name]["status"])
            + " with "
            + str(chat_replies[name]["body"])[:300]
            + "."
        )
    reply = chat_replies["standard"]["body"]
    missing = sorted({"question_type", "report_reference", "street_name", "defect_type",
                      "answer", "source_documents", "was_refused"} - set(reply))
    assert len(missing) == 0, (
        "The assistant's reply carries no "
        + str(missing)
        + ", so what it answered and where from cannot be read."
    )
    assert (reply["was_refused"] is False and reply["question_type"] in config.CHAT_ANSWERED_TYPES
            and reply["report_reference"] is None and reply["street_name"] is None), (
        "Asking what the footway defect standard asks for was answered as "
        + str(reply["question_type"])
        + " about the report "
        + str(reply["report_reference"])
        + " and the street "
        + str(reply["street_name"])
        + (" and refused" if reply["was_refused"] else "")
        + ". The standard answers it, so the question is answered from the documents, names no "
        "report or street and is not refused."
    )
    folder = {name for name in os.listdir(config.DOCS_DIRECTORY) if name.lower().endswith(".pdf")}
    cited = reply["source_documents"]
    searched = search_documents(knowledge_collection["collection"],
                                [chat_replies["standard_question"]])
    grounding = (searched[0][0].get("document_name")
                 if isinstance(searched, list) and len(searched) > 0
                 and isinstance(searched[0], list) and len(searched[0]) > 0
                 and isinstance(searched[0][0], dict) else None)
    assert len(cited) > 0 and set(cited) <= folder and grounding in cited, (
        "The answer cites "
        + str(cited)
        + ". An answer names the documents it came from, each of them in the document folder, "
        "and among them the one holding the passage search_documents finds closest to the "
        "question, "
        + repr(grounding)
        + "; an answer that cannot is an answer from general knowledge."
    )
    assert not set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS), (
        "The answer cites "
        + str(sorted(set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS)))
        + ", which the assistant never quotes: it records past decisions on named reports, not "
        "what the standards say."
    )
    assert len(reply["answer"].strip()) > 0 and reply["answer"] != config.CHAT_REFUSAL_MESSAGE, (
        "The answer given was "
        + repr(reply["answer"][:120])
        + ", which answers nothing."
    )
    turns = chat_replies["turns"]["body"]
    missing = sorted({"turn_id", "username", "user_role", "question_text", "question_type",
                      "report_reference", "street_name", "defect_type", "answer_text",
                      "source_documents", "closest_distance", "was_refused"}
                     - set(turns[0] if len(turns) > 0 else {}))
    assert len(turns) > 0 and len(missing) == 0, (
        "The Street Care Coordinator's chat turns read back as "
        + str(len(turns))
        + " turns"
        + ("" if len(turns) == 0 else ", the first without " + str(missing))
        + ", where each turn read back carries its identifier, who asked and in which role, "
        "the question, its type, its report, street and defect type, the answer, the "
        "documents it cites, its closest distance and whether it was refused."
    )
    asked = [turn for turn in turns if turn["question_text"] == chat_replies["standard_question"]
             and turn["turn_id"] > chat_replies["latest_turn_before"]]
    assert len(asked) > 0, (
        "The Street Care Coordinator's latest turns read back as "
        + str([turn["question_text"] for turn in turns])
        + ", which do not include the question just answered, so the audit trail and the chat "
        "page lose it."
    )
    latest = asked[0]
    assert (latest["question_type"] == reply["question_type"] and latest["was_refused"] is False
            and latest["answer_text"] == reply["answer"]
            and latest["report_reference"] is None
            and latest["username"] == chat_replies["asker"]
            and latest["user_role"] == config.ROLE_STREET_CARE_COORDINATOR), (
        "The stored turn is "
        + str({key: latest[key] for key in ["username", "user_role", "question_type",
                                            "report_reference", "was_refused"]})
        + ", which is not who asked or how the question was just answered."
    )
    assert latest["closest_distance"] <= config.CHAT_MAX_RELEVANCE_DISTANCE, (
        "An answered question is stored with its closest passage "
        + str(latest["closest_distance"])
        + " away, beyond the "
        + str(config.CHAT_MAX_RELEVANCE_DISTANCE)
        + " within which an answer counts as found in the documents."
    )
    assert all(name in latest["source_documents"] for name in cited), (
        "The stored turn cites "
        + repr(latest["source_documents"][:300])
        + " where the answer cited "
        + str(cited)
        + ". The audit trail shows where each answer came from."
    )
    client = service_client["client"]
    adviser, coordinator = staff["adviser"], staff["coordinator"]
    own = client.get("/chat/turns", params={"turn_limit": config.CHAT_AUDIT_MAX_ROWS},
                     headers=adviser["headers"])
    audit = client.get("/audit/chat-turns", params={"turn_limit": config.CHAT_AUDIT_MAX_ROWS},
                       headers=coordinator["headers"])
    own_turns = own.json() if own.status_code == 200 and isinstance(own.json(), list) else None
    audit_turns = (audit.json() if audit.status_code == 200 and isinstance(audit.json(), list)
                   else None)
    assert (own_turns is not None and audit_turns is not None
            and all(turn.get("username") == adviser["username"] for turn in own_turns)
            and latest["turn_id"] not in [turn.get("turn_id") for turn in own_turns]
            and latest["turn_id"] in [turn.get("turn_id") for turn in audit_turns]), (
        "A Customer Services Adviser reading their own chat turns got "
        + str(own.status_code)
        + " with "
        + own.text[:200]
        + ", and a Street Care Coordinator reading the audit trail got "
        + str(audit.status_code)
        + " with "
        + audit.text[:200]
        + ". The chat page lists the asker's own turns alone, never another user's question, "
        "while the audit trail lists everyone's latest turns, the coordinator's question among "
        "them."
    )
    short = client.post("/chat/ask", json={"question": "ok"}, headers=adviser["headers"])
    assert short.status_code == 422, (
        "Asking the street care assistant a question of two characters was answered "
        + str(short.status_code)
        + " with "
        + short.text[:200]
        + ". The request's schema refuses it with 422 before the question reaches the language "
        "model."
    )


def test_chat_memory_holds_own_last_exchanges(chat_replies, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.conversation_memory import recall_conversation
    from config import config
    from persistence.database import session_factory
    from persistence.models import Base
    from persistence.repository import insert_chat_turn

    assert chat_replies["problem"] is None, chat_replies["problem"]
    standard = chat_replies["standard"]
    assert standard["status"] == 200 and isinstance(standard["body"], dict), (
        "Asking the street care assistant what the footway defect standard asks for returned "
        + str(standard["status"])
        + " with "
        + str(standard["body"])[:300]
        + ", so there is no exchange for the next question to remember."
    )
    reply = standard["body"]
    follow_up = "And what does it ask the reporter to send with the report?"
    with session_factory() as session:
        remembered = recall_conversation(session, chat_replies["asker"], follow_up)
        someone_else = recall_conversation(session, chat_replies["someone_else"], follow_up)
    assert all(isinstance(memory, dict)
               and isinstance(memory.get("history_text"), str)
               and isinstance(memory.get("search_text"), str)
               for memory in [remembered, someone_else]), (
        "recall_conversation returned "
        + str(remembered)[:200]
        + ", where it returns the history_text the next prompt is told and the search_text "
        "searched for it."
    )
    assert (chat_replies["standard_question"] in remembered["history_text"]
            and reply["answer"] in remembered["history_text"]), (
        "The conversation recalled for the Street Care Coordinator's next question does not "
        "hold the question just asked and its answer, so a follow-up such as "
        + repr(follow_up)
        + " would be read with no idea what it follows."
    )
    assert (follow_up in remembered["search_text"]
            and chat_replies["standard_question"] in remembered["search_text"]), (
        "The text searched for the follow-up is "
        + repr(remembered["search_text"][:200])
        + ". It carries the new question and the question asked just before it."
    )
    assert (chat_replies["standard_question"] not in someone_else["history_text"]
            and chat_replies["standard_question"] not in someone_else["search_text"]), (
        "The conversation recalled for a Customer Services Adviser carries the Street Care "
        "Coordinator's question "
        + repr(chat_replies["standard_question"])
        + ". Each user's memory is their own last exchanges."
    )
    planted_turns = [
        (config.CHAT_TYPE_DEFECT_STANDARD, "What does the carriageway standard ask for first?",
         "A photograph of the defect is attached to the report, as the carriageway standard "
         "states.", None, None, None),
        (config.CHAT_TYPE_REPORT_STATUS, "What is the status of SC0500001?",
         "REV-PLANTED00001, report SC0500001 revision 1 (Footway) on High Street, opened by "
         "grace: repair ordered.", "SC0500001", None, None),
        (config.CHAT_TYPE_STREET_CARE_RULES, "Who decides a review?",
         "A Highways Inspector appointed for the street's district, who did not open the "
         "review, decides it.", None, None, None),
        (config.CHAT_TYPE_STREET_FIGURE,
         "How many footway defects have been reported on Hawthorn Close?",
         "27 footway defects have been reported on Hawthorn Close.", None, "Hawthorn Close",
         config.FOOTWAY),
    ]
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_chat.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as session:
        for (question_type, question_text, answer_text, report_reference, street_name,
             defect_type) in planted_turns:
            insert_chat_turn(session, {
                "username": chat_replies["asker"],
                "user_role": config.ROLE_STREET_CARE_COORDINATOR,
                "question_text": question_text, "question_type": question_type,
                "report_reference": report_reference, "street_name": street_name,
                "defect_type": defect_type, "answer_text": answer_text, "source_documents": "",
                "closest_distance": 0.5, "was_refused": False})
        recalled = recall_conversation(session, chat_replies["asker"], "And on Littleworth?")
    planted_engine.dispose()
    assert isinstance(recalled, dict) and isinstance(recalled.get("history_text"), str), (
        "recall_conversation returned "
        + str(recalled)[:200]
        + " after four planted exchanges, where it returns the history_text of the last ones."
    )
    history = recalled["history_text"]
    kept = planted_turns[-config.CHAT_HISTORY_TURN_LIMIT:]
    dropped = planted_turns[:-config.CHAT_HISTORY_TURN_LIMIT]
    positions = [history.find(turn[1]) for turn in kept]
    assert (min(positions) >= 0 and positions == sorted(positions)
            and all(turn[1] not in history for turn in dropped)), (
        "After four planted exchanges, the conversation recalled is "
        + repr(history[:500])
        + ". It holds the last "
        + str(config.CHAT_HISTORY_TURN_LIMIT)
        + " exchanges, oldest first, and nothing before them."
    )
    carried = [fact for fact in ["REV-PLANTED00001", "grace", "27 footway defects"]
               if fact in history]
    record_turns = [turn for turn in kept if turn[0] in config.CHAT_RECORD_TYPES]
    assert (len(carried) == 0
            and history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY) == len(record_turns)
            and all(turn[2] in history for turn in kept if turn not in record_turns)), (
        "After a Street Care Coordinator was told a report's status and a street figure, the "
        "conversation recalled for the next question carries "
        + str(carried)
        + " and "
        + str(history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY))
        + " times "
        + repr(config.CHAT_RECORD_ANSWER_IN_HISTORY)
        + ". The recalled conversation goes into the next prompt, and nothing from the records "
        "is ever put in front of the language model: an answer from the records is replaced by "
        "those words, while an answer from the documents is replayed as given."
    )


def test_adviser_hears_status_and_resident_update(saved_run, staff, live_files, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.record_answer import answer_report_record
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun, Base, ReportReview
    from tools.report_data_tools import describe_record, fetch_report

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    reference = saved_run["report_reference"]
    described = describe_record(fetch_report(reports, reference))
    defect_types = described[config.DEFECT_TYPE_COLUMN]
    if described[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT:
        defect_types = config.CHAT_LINKED_DEFECT_TEMPLATE.format(
            defect_type=described[config.DEFECT_TYPE_COLUMN],
            linked_defect_type=described[config.LINKED_DEFECT_TYPE_COLUMN])
    status_lines = {status: config.CHAT_ADVISER_STATUS_TEMPLATE.format(
        report_reference=reference, defect_types=defect_types,
        street_name=described[config.STREET_NAME_COLUMN],
        status=config.REPORT_STATUS_PHRASES[status])
        for status in [config.REVIEW_STATUS_AWAITING, config.REVIEW_STATUS_REPAIR_ORDERED]}
    record = saved_run["final_record"]["body"][0]
    ordered_location = describe_record(record.get("document_details") or {}).get(
        config.LOCATION_DESCRIPTION_COLUMN)
    ordered_opening = (status_lines[config.REVIEW_STATUS_REPAIR_ORDERED] + " "
                       + config.CHAT_ORDERED_REPAIR_TEMPLATE.format(
                           location_description=ordered_location, resident_update=""))
    at_gate = str(saved_run["heard_at_gate"]["adviser_status"])
    after = str(saved_run["heard_after"]["adviser_status"])
    told_update = after[len(ordered_opening):] if after.startswith(ordered_opening) else ""
    stored_update = " ".join(str(record.get("resident_update_text") or "").split())
    assert (at_gate == status_lines[config.REVIEW_STATUS_AWAITING]
            and after.startswith(ordered_opening) and len(told_update.split()) >= 3
            and told_update == " ".join(told_update.split()) and told_update in stored_update
            and "HEADLINE:" not in told_update and "UPDATE:" not in told_update), (
        "A Customer Services Adviser asking about "
        + reference
        + " was told "
        + repr(at_gate[:300])
        + " while the review waited and "
        + repr(after[:500])
        + " after the repair was ordered. An adviser hears the report's defect types, street "
        "and status, first "
        + repr(status_lines[config.REVIEW_STATUS_AWAITING])
        + " and nothing more, and then "
        + repr(ordered_opening)
        + " followed by the resident update the order wrote, on one line, so the caller hears "
        "what the resident was told."
    )
    for moment, answer in [
            ("while the review waited", saved_run["heard_at_gate"]["adviser_prediction"]),
            ("after the repair was ordered", saved_run["heard_after"]["adviser_prediction"])]:
        assert answer == config.OUTCOME_NOT_PREDICTED_LINE, (
            "A Customer Services Adviser asking whether "
            + reference
            + " will come back "
            + moment
            + " was told "
            + repr(str(answer)[:300])
            + " where they hear exactly "
            + repr(config.OUTCOME_NOT_PREDICTED_LINE)
            + ". Customer services is never told a prediction."
        )
    with session_factory() as session:
        stored_review = session.get(ReportReview, saved_run["at_gate"]["review_id"])
        stored_run = session.get(AgentRun, saved_run["at_gate"]["run_id"])
        assert stored_review is not None and stored_run is not None, (
            "The review "
            + saved_run["at_gate"]["review_id"]
            + " or its run "
            + saved_run["at_gate"]["run_id"]
            + " is not in the database, where every review and run is stored."
        )
        review_values = {column.name: getattr(stored_review, column.name)
                         for column in ReportReview.__table__.columns}
        run_values = {column.name: getattr(stored_run, column.name)
                      for column in AgentRun.__table__.columns}
    others = [fetch_report(reports, report_reference)
              for report_reference in reports[config.REPORT_REFERENCE_COLUMN]
              if report_reference != reference]
    revised = [other for other in others if other[config.REVISION_NUMBER_COLUMN] >= 2][:1]
    linked = [other for other in others
              if other[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT
              and other not in revised][:1]
    unlinked = [other for other in others
                if other[config.LINKED_DEFECT_TYPE_COLUMN] == config.NO_LINKED_DEFECT
                and other not in revised][:5]
    assert len(revised) == 1 and len(linked) == 1 and len(unlinked) == 5, (
        "The open reports hold "
        + str(len(revised))
        + " revised, "
        + str(len(linked))
        + " linked and "
        + str(len(unlinked))
        + " unlinked reports besides the one reviewed, where the supplied files hold a revised "
        "report, a linked one and many unlinked."
    )
    ordered = config.REVIEW_STATUS_REPAIR_ORDERED
    inspected = config.REVIEW_STATUS_INSPECTION_REQUESTED
    referred, awaiting = config.REVIEW_STATUS_REFERRED, config.REVIEW_STATUS_AWAITING
    cases = [(unlinked[0], ordered, 0), (linked[0], ordered, 0), (unlinked[1], inspected, 0),
             (unlinked[2], referred, 0), (unlinked[3], awaiting, 0), (unlinked[4], None, 0),
             (revised[0], ordered, 1)]
    planted_update = ("A repair has been ordered for the defect you reported.\nThe crew will "
                      "make the  surface safe.")
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_reviews.db"))
    Base.metadata.create_all(planted_engine)
    told = []
    with sessionmaker(bind=planted_engine)() as planted:
        for position, (picked, status, revisions_back) in enumerate(cases, start=1):
            if status is None:
                continue
            picked_reference = picked[config.REPORT_REFERENCE_COLUMN]
            document_details = None
            resident_update_text = None
            if status == ordered:
                document_details = {
                    config.REPORT_REFERENCE_COLUMN: picked_reference,
                    config.LOCATION_DESCRIPTION_COLUMN: "beside bus stop " + str(position)}
                resident_update_text = ("HEADLINE: Repair ordered for " + picked_reference
                                        + "\n\nUPDATE: " + planted_update)
            planted.add(AgentRun(**dict(
                run_values, run_id="planted-run-" + str(position),
                thread_id="planted-thread-" + str(position), report_reference=picked_reference)))
            planted.add(ReportReview(**dict(
                review_values, review_id="REV-PLANTED" + str(position).zfill(5),
                run_id="planted-run-" + str(position), report_reference=picked_reference,
                revision_number=int(picked[config.REVISION_NUMBER_COLUMN]) - revisions_back,
                defect_type=picked[config.DEFECT_TYPE_COLUMN],
                linked_defect_type=picked[config.LINKED_DEFECT_TYPE_COLUMN],
                street_name=picked[config.STREET_NAME_COLUMN], status=status,
                document_details=document_details, resident_update_text=resident_update_text)))
        planted.flush()
        for position, (picked, status, revisions_back) in enumerate(cases, start=1):
            picked_reference = picked[config.REPORT_REFERENCE_COLUMN]
            picked_described = describe_record(picked)
            picked_types = picked_described[config.DEFECT_TYPE_COLUMN]
            if picked_described[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT:
                picked_types = config.CHAT_LINKED_DEFECT_TEMPLATE.format(
                    defect_type=picked_described[config.DEFECT_TYPE_COLUMN],
                    linked_defect_type=picked_described[config.LINKED_DEFECT_TYPE_COLUMN])
            heard_status = config.REPORT_STATUS_AWAITING_REVIEW
            if status is not None and revisions_back == 0:
                heard_status = status
            expected = config.CHAT_ADVISER_STATUS_TEMPLATE.format(
                report_reference=picked_reference, defect_types=picked_types,
                street_name=picked_described[config.STREET_NAME_COLUMN],
                status=config.REPORT_STATUS_PHRASES[heard_status])
            if heard_status == ordered:
                expected = expected + " " + config.CHAT_ORDERED_REPAIR_TEMPLATE.format(
                    location_description="beside bus stop " + str(position),
                    resident_update=" ".join(planted_update.split()))
            told.append((
                "a Customer Services Adviser asking about "
                + picked_reference
                + " with a planted review "
                + str(status)
                + (" of an earlier revision" if revisions_back else ""),
                answer_report_record(planted, staff["adviser"], reports,
                                     config.CHAT_TYPE_REPORT_STATUS, picked_reference),
                expected))
        unreviewed = unlinked[4][config.REPORT_REFERENCE_COLUMN]
        told.extend([
            ("a Customer Services Adviser naming no report",
             answer_report_record(planted, staff["adviser"], reports,
                                  config.CHAT_TYPE_REPORT_STATUS, None),
             config.CHAT_REPORT_NOT_NAMED_MESSAGE),
            ("a Customer Services Adviser naming SC0099999, which is not an open report",
             answer_report_record(planted, staff["adviser"], reports,
                                  config.CHAT_TYPE_REPORT_STATUS, "SC0099999"),
             config.CHAT_NO_REPORT_TEMPLATE.format(report_reference="SC0099999")),
            ("a Customer Services Adviser asking how long " + unreviewed + " will take",
             answer_report_record(planted, staff["adviser"], reports,
                                  config.CHAT_TYPE_OUTCOME_PREDICTION, unreviewed),
             config.OUTCOME_NOT_PREDICTED_LINE),
        ])
    planted_engine.dispose()
    for meaning, answer, expected in told:
        assert answer == expected, (
            "answer_report_record told "
            + meaning
            + " "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". An adviser hears a report's defect types, street and status at its latest "
            "revision, and once its repair is ordered where on the street the repair is and "
            "what the resident was told - never a prediction."
        )


def test_team_hears_review_record_and_decision(saved_run, staff, live_files, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.record_answer import answer_report_record
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun, Base, ReportReview
    from tools.report_data_tools import describe_record, fetch_report

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    reports = live_files["reports"]
    reference = saved_run["report_reference"]
    described = describe_record(fetch_report(reports, reference))
    defect_types = described[config.DEFECT_TYPE_COLUMN]
    if described[config.LINKED_DEFECT_TYPE_COLUMN] != config.NO_LINKED_DEFECT:
        defect_types = config.CHAT_LINKED_DEFECT_TEMPLATE.format(
            defect_type=described[config.DEFECT_TYPE_COLUMN],
            linked_defect_type=described[config.LINKED_DEFECT_TYPE_COLUMN])
    record = saved_run["final_record"]["body"][0]
    decider = saved_run["decider"]
    review_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id=saved_run["at_gate"]["review_id"], report_reference=reference,
        revision_number=saved_run["revision_number"], defect_types=defect_types,
        street_name=described[config.STREET_NAME_COLUMN],
        coordinator_id=saved_run["opener"]["username"],
        status=config.REPORT_STATUS_PHRASES[config.REVIEW_STATUS_REPAIR_ORDERED])
    figures_line = config.CHAT_FIGURES_TEMPLATE.format(
        repeat_likelihood=format(record["repeat_likelihood"],
                                 "." + str(config.LIKELIHOOD_DECIMALS) + "f"),
        likelihood_band=record["likelihood_band"], crew_hours_text=record["crew_hours_text"])
    decided_line = config.CHAT_DECIDED_TEMPLATE.format(
        decided_by=decider["username"], decided_by_role=config.ROLE_HIGHWAYS_INSPECTOR,
        decided_by_district=decider["district"], decision=config.DECISION_ORDER,
        decision_reason=saved_run["decision_reason"])
    waiting_line = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
        recommended_decision=saved_run["at_gate"]["recommended_decision"])
    team_lines = str(saved_run["heard_after"]["team"]).split("\n")
    waiting_lines = str(saved_run["heard_at_gate"]["team"]).split("\n")
    assert (team_lines[0] == review_line and figures_line in team_lines
            and decided_line in team_lines and waiting_line in waiting_lines), (
        "The street care team asking about "
        + reference
        + " was told "
        + repr(str(saved_run["heard_at_gate"]["team"])[:300])
        + " while the review waited and "
        + repr(str(saved_run["heard_after"]["team"])[:500])
        + " after the repair was ordered. Highways Inspectors and Street Care Coordinators "
        "hear, in the configuration's words, the review "
        + repr(review_line[:160])
        + " first, its figures "
        + repr(figures_line[:160])
        + ", what it recommends while it waits and the decision once taken."
    )
    with session_factory() as session:
        stored_review = session.get(ReportReview, saved_run["at_gate"]["review_id"])
        stored_run = session.get(AgentRun, saved_run["at_gate"]["run_id"])
        assert stored_review is not None and stored_run is not None, (
            "The review "
            + saved_run["at_gate"]["review_id"]
            + " or its run "
            + saved_run["at_gate"]["run_id"]
            + " is not in the database, where every review and run is stored."
        )
        review_values = {column.name: getattr(stored_review, column.name)
                         for column in ReportReview.__table__.columns}
        run_values = {column.name: getattr(stored_run, column.name)
                      for column in AgentRun.__table__.columns}
    others = [fetch_report(reports, report_reference)
              for report_reference in reports[config.REPORT_REFERENCE_COLUMN]
              if report_reference != reference]
    revised = [other for other in others if other[config.REVISION_NUMBER_COLUMN] >= 2][:1]
    unlinked = [other for other in others
                if other[config.LINKED_DEFECT_TYPE_COLUMN] == config.NO_LINKED_DEFECT
                and other not in revised][:2]
    assert len(revised) == 1 and len(unlinked) == 2, (
        "The open reports hold "
        + str(len(revised))
        + " revised and "
        + str(len(unlinked))
        + " unlinked reports besides the one reviewed, where the supplied files hold a revised "
        "report and many unlinked."
    )
    picked, earlier = unlinked[0], revised[0]
    picked_reference = picked[config.REPORT_REFERENCE_COLUMN]
    earlier_reference = earlier[config.REPORT_REFERENCE_COLUMN]
    unreviewed = unlinked[1][config.REPORT_REFERENCE_COLUMN]
    finding = {"document_id": "DOC-990010", "document_type": config.RESIDENT_DESCRIPTION,
               "sentence": "A sentence planted for the checks.", "requirements_borne_on": []}
    planted_review = {
        "review_id": "REV-PLANTED00001", "run_id": "planted-run-1",
        "report_reference": picked_reference,
        "revision_number": int(picked[config.REVISION_NUMBER_COLUMN]),
        "defect_type": picked[config.DEFECT_TYPE_COLUMN],
        "linked_defect_type": picked[config.LINKED_DEFECT_TYPE_COLUMN],
        "street_name": picked[config.STREET_NAME_COLUMN], "status": config.REVIEW_STATUS_AWAITING,
        "requirements": [{"defect_type": picked[config.DEFECT_TYPE_COLUMN],
                          "standard_document": config.DEFECT_STANDARD_DOCUMENTS[
                              picked[config.DEFECT_TYPE_COLUMN]],
                          "requirement_kind": config.LOCATION_FIXED_REQUIREMENT,
                          "requirement_name": "Location fixed",
                          "requirement_sentence": "A sentence planted for the checks.",
                          "result": config.REQUIREMENT_MET, "blank_answers": []}],
        "assessment": config.DECISION_REFER,
        "decisions_open": config.DECISIONS_OPEN[config.DECISION_REFER],
        "coordination_triggers": [config.TRAFFIC_SENSITIVE_TRIGGER],
        "live_permits": ["SWP0951301", "SWP0951302"], "conflicting_permits": ["SWP0951302"],
        "evidence": [dict(finding, sentence_type=config.MEASUREMENT),
                     dict(finding, sentence_type=config.MEASUREMENT),
                     dict(finding, sentence_type=config.HAZARD)],
        "disagreements": [{"document_id": "DOC-990010", "sentence": finding["sentence"],
                           "stated_readings": {}, "report_readings": {},
                           "flag": config.DISAGREES_WITH_FORM}],
        "repeat_likelihood": 0.37, "likelihood_band": config.MEDIUM_LIKELIHOOD_BAND,
        "crew_hours_text": "planted crew hours", "recommended_decision": config.DECISION_REFER,
        "order_document_text": None, "document_details": None, "resident_update_text": None,
        "coordination_brief": None}
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_team.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as planted:
        planted.add(AgentRun(**dict(run_values, run_id="planted-run-1",
                                    thread_id="planted-thread-1",
                                    report_reference=picked_reference)))
        planted.add(ReportReview(**dict(review_values, **planted_review)))
        planted.add(AgentRun(**dict(run_values, run_id="planted-run-2",
                                    thread_id="planted-thread-2",
                                    report_reference=earlier_reference)))
        planted.add(ReportReview(**dict(
            review_values, review_id="REV-PLANTED00002", run_id="planted-run-2",
            report_reference=earlier_reference,
            revision_number=int(earlier[config.REVISION_NUMBER_COLUMN]) - 1,
            defect_type=earlier[config.DEFECT_TYPE_COLUMN],
            linked_defect_type=earlier[config.LINKED_DEFECT_TYPE_COLUMN],
            street_name=earlier[config.STREET_NAME_COLUMN],
            status=config.REVIEW_STATUS_REFERRED)))
        planted.flush()
        heard = str(answer_report_record(planted, staff["coordinator"], reports,
                                         config.CHAT_TYPE_REPORT_STATUS, picked_reference))
        told = [
            ("about " + earlier_reference + ", whose review is of an earlier revision",
             answer_report_record(planted, staff["coordinator"], reports,
                                  config.CHAT_TYPE_OUTCOME_PREDICTION, earlier_reference),
             config.CHAT_NO_REVIEW_TEMPLATE.format(report_reference=earlier_reference)),
            ("about " + unreviewed + ", which nobody has reviewed",
             answer_report_record(planted, staff["coordinator"], reports,
                                  config.CHAT_TYPE_REPORT_STATUS, unreviewed),
             config.CHAT_NO_REVIEW_TEMPLATE.format(report_reference=unreviewed)),
            ("naming no report",
             answer_report_record(planted, staff["coordinator"], reports,
                                  config.CHAT_TYPE_REPORT_STATUS, None),
             config.CHAT_REPORT_NOT_NAMED_MESSAGE),
            ("naming SC0099999, which is not an open report",
             answer_report_record(planted, staff["coordinator"], reports,
                                  config.CHAT_TYPE_REPORT_STATUS, "SC0099999"),
             config.CHAT_NO_REPORT_TEMPLATE.format(report_reference="SC0099999"))]
    planted_engine.dispose()
    lines = heard.split("\n")
    first_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id="REV-PLANTED00001", report_reference=picked_reference,
        revision_number=int(picked[config.REVISION_NUMBER_COLUMN]),
        defect_types=describe_record(picked)[config.DEFECT_TYPE_COLUMN],
        street_name=picked[config.STREET_NAME_COLUMN], coordinator_id=run_values["coordinator_id"],
        status=config.REPORT_STATUS_PHRASES[config.REVIEW_STATUS_AWAITING])
    planted_figures = config.CHAT_FIGURES_TEMPLATE.format(
        repeat_likelihood=format(0.37, "." + str(config.LIKELIHOOD_DECIMALS) + "f"),
        likelihood_band=config.MEDIUM_LIKELIHOOD_BAND, crew_hours_text="planted crew hours")
    planted_waiting = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
        recommended_decision=config.DECISION_REFER)
    screening_opening = config.CHAT_SCREENING_TEMPLATE.split("{requirements}")[0].format(
        assessment=config.DECISION_REFER)
    evidence_parts = config.CHAT_EVIDENCE_TEMPLATE.split("{sentence_types}")
    marked = config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference="SWP0951302")
    unmarked = config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference="SWP0951301")
    assert (lines[0] == first_line and planted_figures in lines and planted_waiting in lines
            and any(line.startswith(screening_opening)
                    and config.TRAFFIC_SENSITIVE_TRIGGER in line for line in lines)
            and any(marked in line and "SWP0951301" in line and unmarked not in line
                    for line in lines)
            and any(line.startswith(evidence_parts[0].format(sentence_count=3))
                    and line.endswith(evidence_parts[1].format(flag_count=1))
                    for line in lines)), (
        "A Street Care Coordinator asking about "
        + picked_reference
        + ", whose planted review waits with a coordination trigger, two street works permits "
        "live on the street, one of them conflicting, three statement sentences read and one "
        "flagged, was told "
        + repr(heard[:700])
        + ". The team hears the review first, "
        + repr(first_line)
        + ", then its screening with the triggers that apply, the permits live on the street "
        "with the conflicting one marked "
        + repr(marked)
        + ", its figures, the statement sentences read and flagged, and what the review "
        "recommends while it waits, each in the configuration's words."
    )
    for meaning, answer, expected in told:
        assert answer == expected, (
            "A Street Care Coordinator asking "
            + meaning
            + " was told "
            + repr(str(answer)[:300])
            + " where the answer is exactly "
            + repr(expected)
            + ". The team hears the review of a report's latest revision only, and is told so "
            "when there is none, when no report is named and when the report is not open."
        )


def test_street_figures_need_five_reports(processed_frames):
    import pandas as pd

    from chat.record_answer import answer_street_figure
    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    template = processed_frames[config.CLEAN_REPORT_HISTORY_EVAL_FILE].head(1)
    planted_rows = []
    footway, vegetation = config.FOOTWAY, config.VEGETATION
    carriageway, drainage = config.CARRIAGEWAY, config.DRAINAGE
    hawthorn, littleworth, kiln, mill = "Hawthorn Close", "Littleworth", "Kiln Lane", "Mill Lane"
    for street_name, defect_type, linked_type, numbers in [
            (hawthorn, footway, config.NO_LINKED_DEFECT, [1, 2, 3, 4, 5, 1]),
            (littleworth, footway, config.NO_LINKED_DEFECT, [11, 12, 13, 14, 11, 12]),
            (hawthorn, vegetation, config.NO_LINKED_DEFECT, [21, 22, 23, 24, 25, 26, 27]),
            (kiln, carriageway, config.NO_LINKED_DEFECT, [41, 42, 43, 44]),
            (kiln, drainage, carriageway, [45, 46, 47]),
            (mill, config.STREET_LIGHTING, config.NO_LINKED_DEFECT, [51, 52, 53, 54, 55])]:
        for number in numbers:
            row = template.copy()
            row[config.STREET_NAME_COLUMN] = street_name
            row[config.DEFECT_TYPE_COLUMN] = defect_type
            row[config.LINKED_DEFECT_TYPE_COLUMN] = linked_type
            row[config.REPORT_REFERENCE_COLUMN] = "SC0" + str(900000 + number)
            planted_rows.append(row)
    planted_history = pd.concat(planted_rows, ignore_index=True)
    figure = config.CHAT_STREET_FIGURE_TEMPLATE
    figure_cases = [
        ("footway on Hawthorn Close, six rows of five reports", hawthorn, footway,
         figure.format(report_count=5, defect_type=footway.lower(), street_name=hawthorn)),
        ("footway on Littleworth, six rows of four reports", littleworth, footway,
         config.GROUP_SIZE_LINE),
        ("vegetation on Hawthorn Close, seven reports", hawthorn, vegetation,
         figure.format(report_count=7, defect_type=vegetation.lower(), street_name=hawthorn)),
        ("carriageway on Kiln Lane, four reports and three drainage reports linked to it", kiln,
         carriageway, config.GROUP_SIZE_LINE),
        ("drainage on Kiln Lane, three reports", kiln, drainage, config.GROUP_SIZE_LINE),
        ("street lighting on Mill Lane, five reports", mill, config.STREET_LIGHTING,
         figure.format(report_count=5, defect_type=config.STREET_LIGHTING.lower(),
                       street_name=mill)),
        ("drainage on Littleworth, none", littleworth, drainage, config.GROUP_SIZE_LINE),
        ("no street", None, footway, config.CHAT_STREET_NOT_NAMED_MESSAGE),
        ("no defect type", hawthorn, None, config.CHAT_STREET_NOT_NAMED_MESSAGE),
    ]
    for meaning, street_name, defect_type, expected in figure_cases:
        answer = answer_street_figure(planted_history, street_name, defect_type)
        assert answer == expected, (
            "Asked about "
            + meaning
            + ", answer_street_figure gave "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". A street figure counts the different reports of that defect type on that "
            "street, never one only linked to the type, and is given only where at least "
            + str(config.GROUP_SIZE_MINIMUM)
            + " reports make it up, for everyone who asks: below that the residents who "
            "reported could be told apart, so the answer is the group-size line."
        )
