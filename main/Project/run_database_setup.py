import os
import secrets
import sys

from sqlalchemy import inspect

from access.password_security import hash_password
from config import config
from persistence.database import database_engine, session_factory
from persistence.models import Base, UserAccount
from persistence.repository import fetch_user_account, insert_user_account


def refuse_stale_tables():
    existing = inspect(database_engine)
    for table in Base.metadata.sorted_tables:
        if not existing.has_table(table.name):
            continue
        found = {(column["name"], column["nullable"]) for column in existing.get_columns(table.name)}
        wanted = {(column.name, column.nullable) for column in table.columns}
        if found != wanted:
            raise SystemExit(f"Table {table.name} is from an earlier version: delete "
                             f"{config.STREETCARE_DATABASE_FILE} and {config.CHECKPOINT_DATABASE_FILE} and run again")


def main():
    os.makedirs(config.DATABASE_DIRECTORY, exist_ok=True)
    refuse_stale_tables()
    Base.metadata.create_all(database_engine)
    print(f"database: {config.STREETCARE_DATABASE_FILE}")
    print(f"  tables: {', '.join(sorted(Base.metadata.tables))}")
    created = present = 0
    with session_factory() as session, session.begin():
        for account in config.SEED_USER_ACCOUNTS:
            if fetch_user_account(session, UserAccount.username == account["username"]) is not None:
                present += 1
                continue
            salt = secrets.token_bytes(config.PASSWORD_SALT_BYTES).hex()
            record = {key: value for key, value in account.items() if key != "password"}
            insert_user_account(session, {**record, "password_salt": salt,
                                          "password_hash": hash_password(account["password"], salt)})
            created += 1
            print(f"  account created  {account['username']:<{config.USERNAME_WIDTH}} "
                  f"{account['display_name']:<{config.DISPLAY_NAME_WIDTH}} {account['role']}")
    print(f"  accounts created: {created}, already present: {present}")


if __name__ == "__main__":
    sys.exit(main())
