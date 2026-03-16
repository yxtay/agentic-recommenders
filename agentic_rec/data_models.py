"""Pydantic models for raw MovieLens dataset tables.

This module defines data models for the raw MovieLens dataset tables:
- movies.dat: Movie titles and genres
- users.dat: User demographics
- ratings.dat: User-movie ratings and timestamps
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class GenderEnum(str, Enum):
    """MovieLens gender categories."""

    FEMALE = "F"
    MALE = "M"


class AgeEnum(int, Enum):
    """MovieLens age bracket categories."""

    UNDER_18 = 1
    AGED_18_24 = 18
    AGED_25_34 = 25
    AGED_35_44 = 35
    AGED_45_49 = 45
    AGED_50_55 = 50
    AGED_56_PLUS = 56


class GenreEnum(str, Enum):
    """MovieLens genre categories."""

    ACTION = "Action"
    ADVENTURE = "Adventure"
    ANIMATION = "Animation"
    CHILDRENS = "Children's"
    COMEDY = "Comedy"
    CRIME = "Crime"
    DOCUMENTARY = "Documentary"
    DRAMA = "Drama"
    FANTASY = "Fantasy"
    FILM_NOIR = "Film-Noir"
    HORROR = "Horror"
    MUSICAL = "Musical"
    MYSTERY = "Mystery"
    ROMANCE = "Romance"
    SCI_FI = "Sci-Fi"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"


class OccupationEnum(int, Enum):
    """MovieLens occupation categories (0-20 mapping)."""

    ADMINISTRATOR = 0
    ARTIST = 1
    DOCTOR = 2
    EDUCATOR = 3
    ENGINEER = 4
    ENTERTAINMENT = 5
    EXECUTIVE = 6
    HEALTHCARE = 7
    HOMEMAKER = 8
    LAWYER = 9
    LIBRARIAN = 10
    MARKETING = 11
    NONE = 12
    OTHER = 13
    PROGRAMMER = 14
    RETIRED = 15
    SALESMAN = 16
    SCIENTIST = 17
    STUDENT = 18
    TECHNICIAN = 19
    WRITER = 20


class RawMovie(BaseModel):
    """Raw MovieLens movie record from movies.dat.

    Attributes:
        movie_id: Unique movie identifier.
        title: Movie title (includes release year in parentheses).
        genres: Pipe-separated list of genres.
    """

    movie_id: str = Field(description="Unique movie identifier")
    title: str = Field(description="Movie title with release year")
    genres: str = Field(description="Pipe-separated genres (e.g., 'Animation|Comedy')")


class RawUser(BaseModel):
    """Raw MovieLens user record from users.dat.

    Attributes:
        user_id: Unique user identifier.
        gender: User gender ('F' for female, 'M' for male).
        age: Age bracket code.
        occupation: Occupation code (0-20).
        zipcode: User's zip code.
    """

    user_id: str = Field(description="Unique user identifier")
    gender: GenderEnum = Field(description="User gender (F/M)")
    age: AgeEnum = Field(description="Age bracket code")
    occupation: int = Field(ge=0, le=20, description="Occupation code (0-20)")
    zipcode: str = Field(description="User's zip code")


class RawRating(BaseModel):
    """Raw MovieLens rating record from ratings.dat.

    Attributes:
        user_id: Unique user identifier.
        movie_id: Unique movie identifier.
        rating: Rating value (1-5 stars).
        timestamp: Unix timestamp of when the rating was given.
    """

    user_id: str = Field(description="Unique user identifier")
    movie_id: str = Field(description="Unique movie identifier")
    rating: int = Field(ge=1, le=5, description="Rating value (1-5 stars)")
    timestamp: int = Field(ge=0, description="Unix timestamp of the rating")

    @field_validator("timestamp", mode="after")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """Validate that timestamp is a reasonable Unix timestamp.

        MovieLens 1M timestamps are from the 1997-2003 period.
        """
        if v < 0 or v > 2000000000:
            raise ValueError(f"Unreasonable timestamp: {v}")
        return v
