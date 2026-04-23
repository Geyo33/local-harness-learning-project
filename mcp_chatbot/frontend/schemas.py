from pydantic import BaseModel, Field

# Not used yet
class UserContext(BaseModel):
    user_id: str = Field(default="User")
    history: list = Field(default=[])
    mcp_servers: list = Field(default=[])