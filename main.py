import uvicorn
from fastapi.middleware.wsgi import WSGIMiddleware

from src.app import app
from web.app import app as WebApp

# code adapted from (Redditinc, n.d.)
app.mount("/dashboard", WSGIMiddleware(WebApp.server))

@app.get("/")
async def root():
    return {"message": "Welcome...!, to ASTAPM API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# end of adapted code
