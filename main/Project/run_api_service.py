import os
import sys

from config import config


def main():
    missing = [(path, producer) for path, producer in config.SERVICE_REQUIRED_FILES if not os.path.exists(path)]
    if missing:
        print("The StreetCare AI service cannot start. Missing:")
        for path, producer in missing:
            print(f"  {path}  (produced by {producer})")
        return 1
    import uvicorn

    from api.routes import service_application
    print(f"Interactive documentation: http://localhost:{config.API_PORT}/docs")
    uvicorn.run(service_application, host=config.API_HOST, port=config.API_PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
