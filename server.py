from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import httpx
from enum import Enum

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]



# OpenRouteService API
ORS_API_KEY = os.environ.get('OPENROUTESERVICE_API_KEY', '')
ORS_BASE_URL = "https://api.openrouteservice.org"

# Create the main app
app = FastAPI(title="Route Optimizer API")

# Create a router with the /api prefix
api_router = APIRouter()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class TransportProfile(str, Enum):
    DRIVING_CAR = "driving-car"
    FOOT_WALKING = "foot-walking"

class Coordinate(BaseModel):
    longitude: float
    latitude: float

class WaypointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WaypointColor(str, Enum):
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    YELLOW = "yellow"
    GRAY = "gray"

class Waypoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    address: str
    coordinates: Coordinate
    status: WaypointStatus = WaypointStatus.PENDING
    note: Optional[str] = None
    color: WaypointColor = WaypointColor.BLUE

class RouteCreate(BaseModel):
    name: str
    start: Waypoint
    end: Waypoint
    waypoints: List[Waypoint] = []
    profile: TransportProfile = TransportProfile.DRIVING_CAR

class RouteUpdate(BaseModel):
    name: Optional[str] = None
    start: Optional[Waypoint] = None
    end: Optional[Waypoint] = None
    waypoints: Optional[List[Waypoint]] = None
    profile: Optional[TransportProfile] = None

class RouteResponse(BaseModel):
    id: str
    name: str
    start: Waypoint
    end: Waypoint
    waypoints: List[Waypoint]
    profile: TransportProfile
    distance: Optional[float] = None
    duration: Optional[float] = None
    optimized_order: Optional[List[int]] = None
    geometry: Optional[dict] = None
    created_at: str
    updated_at: str

class GeocodeResult(BaseModel):
    name: str
    address: str
    coordinates: Coordinate

# ==================== OPENROUTESERVICE FUNCTIONS ====================

async def geocode_address(address: str, country: str = "FR") -> Optional[GeocodeResult]:
    """Geocode an address to coordinates using OpenRouteService, fallback to Nominatim"""
    
    # Try OpenRouteService first
    async with httpx.AsyncClient() as client_http:
        try:
            params = {
                "api_key": ORS_API_KEY,
                "text": address,
                "size": 5,
                "boundary.country": country,
                "layers": "address,venue,street",
            }
            
            response = await client_http.get(
                f"{ORS_BASE_URL}/geocode/search",
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("features") and len(data["features"]) > 0:
                feature = data["features"][0]
                coords = feature["geometry"]["coordinates"]
                props = feature.get("properties", {})
                
                house_number = props.get("housenumber", "")
                street = props.get("street", "")
                name = props.get("name", "")
                label = props.get("label", address)
                
                if house_number and street:
                    display_name = f"{house_number} {street}"
                elif name:
                    display_name = name
                else:
                    display_name = label.split(",")[0] if label else address
                
                return GeocodeResult(
                    name=display_name,
                    address=label,
                    coordinates=Coordinate(longitude=coords[0], latitude=coords[1])
                )
        except Exception as e:
            logger.error(f"ORS Geocoding error: {e}")
        
        # Fallback to Nominatim (OpenStreetMap)
        try:
            logger.info(f"Trying Nominatim fallback for: {address}")
            nominatim_response = await client_http.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                    "countrycodes": country.lower(),
                    "addressdetails": 1,
                },
                headers={"User-Agent": "RouteOptimizer/1.0"},
                timeout=10.0
            )
            nominatim_response.raise_for_status()
            nominatim_data = nominatim_response.json()
            
            if nominatim_data and len(nominatim_data) > 0:
                result = nominatim_data[0]
                addr = result.get("address", {})
                
                # Build display name
                house_number = addr.get("house_number", "")
                road = addr.get("road", "")
                
                if house_number and road:
                    display_name = f"{house_number} {road}"
                else:
                    display_name = result.get("display_name", address).split(",")[0]
                
                return GeocodeResult(
                    name=display_name,
                    address=result.get("display_name", address),
                    coordinates=Coordinate(
                        longitude=float(result["lon"]),
                        latitude=float(result["lat"])
                    )
                )
        except Exception as e:
            logger.error(f"Nominatim Geocoding error: {e}")
        
        return None

async def calculate_route(coordinates: List[List[float]], profile: str) -> dict:
    """Calculate route using OpenRouteService Directions API"""
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(
                f"{ORS_BASE_URL}/v2/directions/{profile}/geojson",
                headers={
                    "Authorization": ORS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "coordinates": coordinates,
                    "instructions": False
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"ORS API error: {response.status_code} - {error_detail}")
                raise HTTPException(status_code=500, detail=f"Route calculation failed: {error_detail}")
            
            return response.json()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Route calculation error: {e}")
            raise HTTPException(status_code=500, detail=f"Route calculation failed: {str(e)}")

async def optimize_route(jobs: List[dict], vehicle: dict) -> dict:
    """Optimize route order using OpenRouteService Optimization API (VROOM)"""
    async with httpx.AsyncClient() as client_http:
        try:
            response = await client_http.post(
                f"{ORS_BASE_URL}/optimization",
                headers={
                    "Authorization": ORS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "jobs": jobs,
                    "vehicles": [vehicle]
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Route optimization error: {e}")
            raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")

# ==================== API ENDPOINTS ====================

@api_router.get("/")
async def root():
    return {"message": "Route Optimizer API"}

@api_router.get("/geocode")
async def geocode(address: str):
    """Geocode an address to coordinates"""
    if not address:
        raise HTTPException(status_code=400, detail="Address is required")
    
    result = await geocode_address(address)
    if result:
        response = result.model_dump()
        # Preserve original address input for display if it contains a number
        original_parts = address.strip().split()
        if original_parts and original_parts[0].isdigit():
            # User entered a street number, preserve it in display
            response["name"] = address.split(",")[0].strip()
            response["address"] = f"{address.split(',')[0].strip()}, {result.address.split(',', 1)[-1].strip() if ',' in result.address else result.address}"
        return response
    raise HTTPException(status_code=404, detail="Address not found")

@api_router.get("/autocomplete")
async def autocomplete_address(text: str):
    """Get address suggestions for autocomplete - ORS + Nominatim fallback"""
    if not text or len(text) < 3:
        return {"suggestions": []}
    
    suggestions = []
    
    async with httpx.AsyncClient() as client_http:
        # Try OpenRouteService first
        try:
            response = await client_http.get(
                f"{ORS_BASE_URL}/geocode/autocomplete",
                params={
                    "api_key": ORS_API_KEY,
                    "text": text,
                    "boundary.country": "FR",
                    "size": 5,
                },
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                coords = feature["geometry"]["coordinates"]
                
                house_number = props.get("housenumber", "")
                street = props.get("street", "")
                name = props.get("name", "")
                label = props.get("label", "")
                
                if house_number and street:
                    display_name = f"{house_number} {street}"
                elif name:
                    display_name = name
                else:
                    display_name = label.split(",")[0] if label else text
                
                suggestions.append({
                    "name": display_name,
                    "address": label,
                    "coordinates": {
                        "longitude": coords[0],
                        "latitude": coords[1]
                    },
                    "source": "ors"
                })
        except Exception as e:
            logger.error(f"ORS Autocomplete error: {e}")
        
        # If less than 3 results, try Nominatim as supplement
        if len(suggestions) < 3:
            try:
                nominatim_response = await client_http.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": text,
                        "format": "json",
                        "limit": 5,
                        "countrycodes": "fr",
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": "RouteOptimizer/1.0"},
                    timeout=5.0
                )
                nominatim_response.raise_for_status()
                nominatim_data = nominatim_response.json()
                
                # Track existing addresses to avoid duplicates
                existing_addresses = {s["address"].lower() for s in suggestions}
                
                for result in nominatim_data:
                    display_name_full = result.get("display_name", "")
                    if display_name_full.lower() in existing_addresses:
                        continue
                    
                    addr = result.get("address", {})
                    house_number = addr.get("house_number", "")
                    road = addr.get("road", "")
                    
                    if house_number and road:
                        display_name = f"{house_number} {road}"
                    else:
                        display_name = display_name_full.split(",")[0]
                    
                    suggestions.append({
                        "name": display_name,
                        "address": display_name_full,
                        "coordinates": {
                            "longitude": float(result["lon"]),
                            "latitude": float(result["lat"])
                        },
                        "source": "nominatim"
                    })
                    
                    if len(suggestions) >= 5:
                        break
                        
            except Exception as e:
                logger.error(f"Nominatim Autocomplete error: {e}")
        
        return {"suggestions": suggestions[:5]}

@api_router.post("/routes", response_model=RouteResponse)
async def create_route(route: RouteCreate):
    """Create a new route"""
    now = datetime.now(timezone.utc).isoformat()
    route_doc = {
        "id": str(uuid.uuid4()),
        "name": route.name,
        "start": route.start.model_dump(),
        "end": route.end.model_dump(),
        "waypoints": [wp.model_dump() for wp in route.waypoints],
        "profile": route.profile.value,
        "distance": None,
        "duration": None,
        "optimized_order": None,
        "geometry": None,
        "created_at": now,
        "updated_at": now
    }
    
    await db.routes.insert_one(route_doc)
    route_doc.pop("_id", None)
    return route_doc

@api_router.get("/routes", response_model=List[RouteResponse])
async def get_routes():
    """Get all routes"""
    routes = await db.routes.find({}, {"_id": 0}).to_list(100)
    return routes

@api_router.get("/routes/{route_id}", response_model=RouteResponse)
async def get_route(route_id: str):
    """Get a specific route"""
    route = await db.routes.find_one({"id": route_id}, {"_id": 0})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route

@api_router.put("/routes/{route_id}", response_model=RouteResponse)
async def update_route(route_id: str, route_update: RouteUpdate):
    """Update a route"""
    existing = await db.routes.find_one({"id": route_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Route not found")
    
    update_data = {}
    if route_update.name is not None:
        update_data["name"] = route_update.name
    if route_update.start is not None:
        update_data["start"] = route_update.start.model_dump()
    if route_update.end is not None:
        update_data["end"] = route_update.end.model_dump()
    if route_update.waypoints is not None:
        update_data["waypoints"] = [wp.model_dump() for wp in route_update.waypoints]
    if route_update.profile is not None:
        update_data["profile"] = route_update.profile.value
    
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    await db.routes.update_one({"id": route_id}, {"$set": update_data})
    
    updated = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return updated

@api_router.delete("/routes/{route_id}")
async def delete_route(route_id: str):
    """Delete a route"""
    result = await db.routes.delete_one({"id": route_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Route not found")
    return {"message": "Route deleted successfully"}

@api_router.post("/routes/{route_id}/calculate", response_model=RouteResponse)
async def calculate_route_directions(route_id: str):
    """Calculate directions for a route"""
    route = await db.routes.find_one({"id": route_id}, {"_id": 0})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # Build coordinates list: start -> waypoints -> end
    coordinates = []
    coordinates.append([route["start"]["coordinates"]["longitude"], route["start"]["coordinates"]["latitude"]])
    
    for wp in route.get("waypoints", []):
        coordinates.append([wp["coordinates"]["longitude"], wp["coordinates"]["latitude"]])
    
    coordinates.append([route["end"]["coordinates"]["longitude"], route["end"]["coordinates"]["latitude"]])
    
    # Calculate route
    result = await calculate_route(coordinates, route["profile"])
    
    # Extract distance and duration
    if result.get("features") and len(result["features"]) > 0:
        properties = result["features"][0].get("properties", {})
        summary = properties.get("summary", {})
        
        update_data = {
            "distance": summary.get("distance", 0),
            "duration": summary.get("duration", 0),
            "geometry": result["features"][0].get("geometry"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        await db.routes.update_one({"id": route_id}, {"$set": update_data})
    
    updated = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return updated

@api_router.post("/routes/{route_id}/optimize", response_model=RouteResponse)
async def optimize_route_order(route_id: str):
    """Optimize the order of waypoints for a route"""
    route = await db.routes.find_one({"id": route_id}, {"_id": 0})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = route.get("waypoints", [])
    if len(waypoints) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 waypoints to optimize")
    
    # Build jobs for optimization
    jobs = []
    for idx, wp in enumerate(waypoints):
        jobs.append({
            "id": idx,
            "location": [wp["coordinates"]["longitude"], wp["coordinates"]["latitude"]],
            "service": 300
        })
    
    # Vehicle with start and end
    vehicle = {
        "id": 0,
        "profile": route["profile"],
        "start": [route["start"]["coordinates"]["longitude"], route["start"]["coordinates"]["latitude"]],
        "end": [route["end"]["coordinates"]["longitude"], route["end"]["coordinates"]["latitude"]]
    }
    
    # Optimize
    result = await optimize_route(jobs, vehicle)
    
    # Extract optimized order
    optimized_order = []
    if result.get("routes") and len(result["routes"]) > 0:
        for step in result["routes"][0].get("steps", []):
            if step.get("type") == "job":
                optimized_order.append(step["job"])
    
    # Reorder waypoints
    if optimized_order:
        new_waypoints = [waypoints[i] for i in optimized_order if i < len(waypoints)]
        update_data = {
            "waypoints": new_waypoints,
            "optimized_order": optimized_order,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        await db.routes.update_one({"id": route_id}, {"$set": update_data})
    
    # Calculate the optimized route
    updated = await db.routes.find_one({"id": route_id}, {"_id": 0})
    
    # Recalculate directions
    coordinates = []
    coordinates.append([updated["start"]["coordinates"]["longitude"], updated["start"]["coordinates"]["latitude"]])
    for wp in updated.get("waypoints", []):
        coordinates.append([wp["coordinates"]["longitude"], wp["coordinates"]["latitude"]])
    coordinates.append([updated["end"]["coordinates"]["longitude"], updated["end"]["coordinates"]["latitude"]])
    
    route_result = await calculate_route(coordinates, updated["profile"])
    
    if route_result.get("features") and len(route_result["features"]) > 0:
        properties = route_result["features"][0].get("properties", {})
        summary = properties.get("summary", {})
        
        final_update = {
            "distance": summary.get("distance", 0),
            "duration": summary.get("duration", 0),
            "geometry": route_result["features"][0].get("geometry"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        await db.routes.update_one({"id": route_id}, {"$set": final_update})
    
    final = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return final

@api_router.post("/routes/{route_id}/waypoints", response_model=RouteResponse)
async def add_waypoint(route_id: str, waypoint: Waypoint):
    """Add a waypoint to a route"""
    route = await db.routes.find_one({"id": route_id})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = route.get("waypoints", [])
    waypoints.append(waypoint.model_dump())
    
    await db.routes.update_one(
        {"id": route_id},
        {"$set": {
            "waypoints": waypoints,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "geometry": None,
            "distance": None,
            "duration": None
        }}
    )
    
    updated = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return updated

@api_router.delete("/routes/{route_id}/waypoints/{waypoint_id}", response_model=RouteResponse)
async def remove_waypoint(route_id: str, waypoint_id: str):
    """Remove a waypoint from a route"""
    route = await db.routes.find_one({"id": route_id})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = [wp for wp in route.get("waypoints", []) if wp["id"] != waypoint_id]
    
    await db.routes.update_one(
        {"id": route_id},
        {"$set": {
            "waypoints": waypoints,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "geometry": None,
            "distance": None,
            "duration": None
        }}
    )
    
    updated = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return updated

@api_router.patch("/routes/{route_id}/waypoints/{waypoint_id}/status", response_model=RouteResponse)
async def update_waypoint_status(route_id: str, waypoint_id: str, status: WaypointStatus):
    """Update waypoint status (completed, failed, skipped)"""
    route = await db.routes.find_one({"id": route_id})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = route.get("waypoints", [])
    updated = False
    
    for wp in waypoints:
        if wp["id"] == waypoint_id:
            wp["status"] = status.value
            updated = True
            break
    
    if not updated:
        raise HTTPException(status_code=404, detail="Waypoint not found")
    
    await db.routes.update_one(
        {"id": route_id},
        {"$set": {
            "waypoints": waypoints,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    result = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return result

class WaypointUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    coordinates: Optional[Coordinate] = None
    status: Optional[WaypointStatus] = None
    note: Optional[str] = None
    color: Optional[WaypointColor] = None

@api_router.patch("/routes/{route_id}/waypoints/{waypoint_id}", response_model=RouteResponse)
async def update_waypoint(route_id: str, waypoint_id: str, update: WaypointUpdate):
    """Update waypoint details (name, address, note, color, status)"""
    route = await db.routes.find_one({"id": route_id})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = route.get("waypoints", [])
    updated = False
    
    for wp in waypoints:
        if wp["id"] == waypoint_id:
            if update.name is not None:
                wp["name"] = update.name
            if update.address is not None:
                wp["address"] = update.address
            if update.coordinates is not None:
                wp["coordinates"] = update.coordinates.model_dump()
            if update.status is not None:
                wp["status"] = update.status.value
            if update.note is not None:
                wp["note"] = update.note
            if update.color is not None:
                wp["color"] = update.color.value
            updated = True
            break
    
    if not updated:
        raise HTTPException(status_code=404, detail="Waypoint not found")
    
    await db.routes.update_one(
        {"id": route_id},
        {"$set": {
            "waypoints": waypoints,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    result = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return result

@api_router.post("/routes/{route_id}/undo", response_model=RouteResponse)
async def undo_last_waypoint_status(route_id: str):
    """Reset the last modified waypoint status to pending"""
    route = await db.routes.find_one({"id": route_id})
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    waypoints = route.get("waypoints", [])
    
    # Find the last waypoint that is not pending and reset it
    for wp in reversed(waypoints):
        if wp.get("status", "pending") != "pending":
            wp["status"] = "pending"
            break
    
    await db.routes.update_one(
        {"id": route_id},
        {"$set": {
            "waypoints": waypoints,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    result = await db.routes.find_one({"id": route_id}, {"_id": 0})
    return result

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
