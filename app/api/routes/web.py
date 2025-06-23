from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="web/templates")


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Serve the search interface page"""
    return templates.TemplateResponse("search.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve the dashboard/index page"""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat interface page"""
    return templates.TemplateResponse("chat.html", {"request": request})
