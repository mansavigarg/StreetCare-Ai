# 8b79ac61-e015-423a-a78c-0534c04ae723-5c75083f-514a-4925-ad31-4fe37ba65c38
https://sonar.server.examly.io/dashboard?id=iamneo-production-2_8b79ac61-e015-423a-a78c-0534c04ae723-5c75083f-514a-4925-ad31-4fe37ba65c38&amp;codeScope=overall

## Docker deployment

Build and run the StreetCare AI dashboard from the `Project` directory:

```bash
cd 8b79ac61-e015-423a-a78c-0534c04ae723-5c75083f-514a-4925-ad31-4fe37ba65c38-main/Project
docker build -t streetcare-ai .
docker run --rm -p 8080:8080 streetcare-ai
```

The container starts both the FastAPI service and Streamlit dashboard. Open
<http://localhost:8080>. Set `PORT` when the deployment platform provides a
different public port. The SQLite databases are container-local; mount
`/app/database` to persistent storage if data must survive restarts.
