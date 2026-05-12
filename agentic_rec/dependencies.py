from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from agentic_rec.repositories.item_repository import ItemRepository
from agentic_rec.repositories.user_repository import UserRepository
from agentic_rec.services.item_service import ItemService
from agentic_rec.services.recommendation_service import RecommendationService
from agentic_rec.services.user_service import UserService


def get_item_repository(request: Request) -> ItemRepository:
    return ItemRepository(request.app.state.items_index)


def get_user_repository(request: Request) -> UserRepository:
    return UserRepository(request.app.state.users_index)


def get_item_service(
    repo: ItemRepository = Depends(get_item_repository),
) -> ItemService:
    return ItemService(repo)


def get_user_service(
    repo: UserRepository = Depends(get_user_repository),
) -> UserService:
    return UserService(repo)


def get_recommendation_service(
    item_repo: ItemRepository = Depends(get_item_repository),
    user_repo: UserRepository = Depends(get_user_repository),
) -> RecommendationService:
    return RecommendationService(item_repo, user_repo)


ItemServiceDep = Annotated[ItemService, Depends(get_item_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
RecServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]
