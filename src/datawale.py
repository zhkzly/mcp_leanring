import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("桌面 TXT 文件统计器")

@mcp.tool()
def count_desktop_txt_files() -> int:
    """Count the number of .txt files on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")
    
    # Count .txt files
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)

# @mcp.prompt


@mcp.tool()
def list_desktop_txt_files() -> str:
    """Get a list of all .txt filenames on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")
    
    # Get all .txt files
    txt_files = list(desktop_path.glob("*.txt"))
    
    # Return the filenames
    if not txt_files:
        return"No .txt files found on desktop."
    
    # Format the list of filenames
    file_list = "\n".join([f"- {file.name}"for file in txt_files])
    return f"Found {len(txt_files)} .txt files on desktransform: translateY(\n{file_list}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()