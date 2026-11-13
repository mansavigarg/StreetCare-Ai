from typing import Annotated

from fastapi import Header, HTTPException, Request

from config import config
from persistence.database import session_factory
from persistence.models import UserAccount
from persistence.repository import fetch_user_account


def authorize_request(request: Request,
                      session_token: Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)] = None):
    account = None
    if session_token:
        with session_factory() as session:
            account = fetch_user_account(session, UserAccount.session_token == session_token)
    if account is None:
        raise HTTPException(401, config.SIGN_IN_REQUIRED_MESSAGE)
    permission = config.PERMISSION_BY_ROUTE[(request.method, request.scope["route"].path)]
    if permission not in config.ROLE_PERMISSIONS[account["role"]]:
        raise HTTPException(403, config.PERMISSION_REFUSED_MESSAGE.format(role=account["role"], permission=permission))
    return account


def authorize_district(signed_in_user, street_name, report_reference):
    if signed_in_user["role"] != config.ROLE_HIGHWAYS_INSPECTOR:
        return None
    district = signed_in_user["district"]
    if street_name not in config.INSPECTION_DISTRICT_STREETS.get(district, []):
        raise HTTPException(403, config.DISTRICT_REFUSED_MESSAGE.format(
            username=signed_in_user["username"], district=district, street_name=street_name,
            report_reference=report_reference))
    return None
