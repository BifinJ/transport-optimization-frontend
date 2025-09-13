"use client";

import React, { useState, useEffect, useRef, FormEvent } from "react";
// Libraries are now loaded dynamically via script tags.

// --- TYPE DEFINITIONS ---
interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
}

// Route data structure based on your API response
interface RouteStop {
  sequence: number;
  type: "pickup" | "delivery";
  package_id: string;
  eta: string;
  coordinates: { lat: number; lng: number };
  address: string;
}

interface VehicleRoute {
  vehicle_id: string;
  driver_name: string;
  stops: RouteStop[];
}

interface RouteData {
  status: string;
  data: {
    routes: VehicleRoute[];
  };
}

// --- SVG ICONS (as React Components) ---

const BotIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {" "}
    <path d="M12 8V4H8" /> <rect width="16" height="12" x="4" y="8" rx="2" />{" "}
    <path d="M2 14h2" /> <path d="M20 14h2" /> <path d="M15 13v2" />{" "}
    <path d="M9 13v2" />{" "}
  </svg>
);

const SendIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {" "}
    <path d="m22 2-7 20-4-9-9-4Z" /> <path d="m22 2-11 11" />{" "}
  </svg>
);

const MapIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {" "}
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>{" "}
    <circle cx="12" cy="10" r="3"></circle>{" "}
  </svg>
);

const ChevronLeftIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m15 18-6-6 6-6" />
  </svg>
);

const ChevronRightIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="m9 18 6-6-6-6" />
  </svg>
);

const CarIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M14 16.94V19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h1.35c.34 0 .68.07 1 .2l2.22 1.11c.98.49 2.17.49 3.15 0l2.22-1.11c.32-.13.66-.2 1-.2H20a2 2 0 0 1 2 2v2.06" />
    <path d="M7 9H5a1 1 0 0 0-1 1v1" />
    <circle cx="7" cy="16" r="2" />
    <circle cx="17" cy="16" r="2" />
  </svg>
);

const BikeIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="18.5" cy="17.5" r="3.5" />
    <circle cx="5.5" cy="17.5" r="3.5" />
    <path d="M15 17.5h-1.5l-3-6-2 3h-2" />
    <path d="m6 14 1-1h1.5" />
    <path d="M15 6h-3l-2 5" />
  </svg>
);

const BusIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M8 6v6" />
    <path d="M16 6v6" />
    <rect width="18" height="12" x="3" y="10" rx="2" />
    <path d="M3 10V8a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v2" />
    <path d="M12 18h.01" />
  </svg>
);

const WalkIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="5" r="1" />
    <path d="M9 20l3-6-4-3-3 6h7" />
    <path d="M15 20l-3-6 4-3 3 6h-4" />
  </svg>
);

const RouteIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m4.93 19.07 1.41-1.41" />
    <path d="m17.66 6.34 1.41-1.41" />
  </svg>
);

const ChatIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

// --- CHAT & ROUTE COMPONENTS ---

const ChatMessage = ({ message }: { message: Message }) => {
  const isUser = message.sender === "user";
  return (
    <div
      className={`flex items-start gap-3 my-4 ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      {!isUser && <BotIcon className="w-5 h-5 text-gray-400 flex-shrink-0" />}
      <div
        className={`px-3 py-2 rounded-lg max-w-md ${
          isUser
            ? "bg-blue-600 text-white rounded-br-none"
            : "bg-gray-700 text-gray-200 rounded-bl-none"
        }`}
      >
        <p className="text-sm">{message.text}</p>
      </div>
    </div>
  );
};

const ChatHistory = ({ messages }: { messages: Message[] }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);
  return (
    <div ref={scrollRef} className="flex-1 p-4 space-y-2 overflow-y-auto">
      {messages.length > 0 ? (
        messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
      ) : (
        <div className="flex flex-col items-center justify-center h-full text-gray-400">
          <BotIcon className="w-12 h-12 mb-2" />
          <p className="text-center text-sm">Start a conversation.</p>
        </div>
      )}
    </div>
  );
};

const ChatInput = ({
  onSendMessage,
}: {
  onSendMessage: (message: string) => void;
}) => {
  const [inputValue, setInputValue] = useState("");
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue.trim());
      setInputValue("");
    }
  };
  return (
    <div className="p-2 bg-gray-900 border-t border-gray-700">
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask anything..."
          className="flex-1 w-full px-3 py-2 text-gray-200 bg-gray-800 border border-gray-700 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
        />
        <button
          type="submit"
          className="p-2 text-white bg-blue-600 rounded-full hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          disabled={!inputValue.trim()}
        >
          <SendIcon className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
};

const RoutePlanner = ({ onFindRoute }: { onFindRoute: () => void }) => {
  const [source, setSource] = useState("123 Main St, New York");
  const [destination, setDestination] = useState("456 Oak Ave, New York");
  const [vehicle, setVehicle] = useState("car");

  const vehicleOptions = [
    { id: "car", label: "Car", icon: <CarIcon className="w-5 h-5" /> },
    { id: "bike", label: "Bike", icon: <BikeIcon className="w-5 h-5" /> },
    { id: "bus", label: "Bus", icon: <BusIcon className="w-5 h-5" /> },
    { id: "walk", label: "Walk", icon: <WalkIcon className="w-5 h-5" /> },
  ];
  return (
    <div className="p-4 space-y-3">
      <div className="space-y-1">
        <label htmlFor="source" className="text-xs font-semibold text-gray-400">
          SOURCE
        </label>
        <input
          id="source"
          type="text"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="e.g., Marine Drive"
          className="w-full p-2 text-sm text-white bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="space-y-1">
        <label
          htmlFor="destination"
          className="text-xs font-semibold text-gray-400"
        >
          DESTINATION
        </label>
        <input
          id="destination"
          type="text"
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
          placeholder="e.g., Lulu Mall"
          className="w-full p-2 text-sm text-white bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="space-y-2">
        <label className="text-xs font-semibold text-gray-400">TRANSPORT</label>
        <div className="grid grid-cols-4 gap-2">
          {vehicleOptions.map((opt) => (
            <button
              key={opt.id}
              onClick={() => setVehicle(opt.id)}
              className={`flex flex-col items-center justify-center p-2 rounded-md transition-colors ${
                vehicle === opt.id
                  ? "bg-blue-600 text-white"
                  : "bg-gray-600 hover:bg-gray-500 text-gray-300"
              }`}
            >
              {opt.icon}
              <span className="text-xs mt-1">{opt.label}</span>
            </button>
          ))}
        </div>
      </div>
      <button
        onClick={onFindRoute}
        className="w-full px-4 py-2 font-semibold text-white bg-green-600 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 text-sm"
      >
        Get Directions
      </button>
    </div>
  );
};

const AssistantPanel = ({ onFindRoute }: { onFindRoute: () => void }) => {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Plan a trip or ask me anything.", sender: "bot" },
  ]);
  const [activeTab, setActiveTab] = useState<"planner" | "chat">("planner");

  const handleSendMessage = (text: string) => {
    setMessages((prev) => [...prev, { id: Date.now(), text, sender: "user" }]);
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, text: `You asked: "${text}".`, sender: "bot" },
      ]);
    }, 1000);
  };

  const TabButton = ({
    label,
    icon,
    isActive,
    onClick,
  }: {
    label: string;
    icon: React.ReactNode;
    isActive: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 text-sm font-medium transition-colors ${
        isActive
          ? "bg-gray-700 text-white"
          : "text-gray-400 hover:bg-gray-700 hover:text-white"
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="flex flex-col h-full bg-gray-800 text-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold whitespace-nowrap">
          Route Assistant
        </h2>
      </div>

      <div className="flex border-b border-gray-700">
        <TabButton
          label="Planner"
          icon={<RouteIcon className="w-4 h-4" />}
          isActive={activeTab === "planner"}
          onClick={() => setActiveTab("planner")}
        />
        <TabButton
          label="Chat"
          icon={<ChatIcon className="w-4 h-4" />}
          isActive={activeTab === "chat"}
          onClick={() => setActiveTab("chat")}
        />
      </div>

      {activeTab === "planner" && (
        <div className="flex-1 overflow-y-auto">
          <RoutePlanner onFindRoute={onFindRoute} />
        </div>
      )}

      {activeTab === "chat" && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatHistory messages={messages} />
          <ChatInput onSendMessage={handleSendMessage} />
        </div>
      )}
    </div>
  );
};

// --- GEOGRAPHICAL MAP COMPONENT ---

const Routing = ({
  mapInstance,
  routeData,
  L,
}: {
  mapInstance: any;
  routeData: RouteData;
  L: any;
}) => {
  useEffect(() => {
    if (!mapInstance || !L || !(L as any).Routing) return;

    const route = routeData.data.routes[0];
    if (!route) return;

    const waypoints = route.stops.map((stop) =>
      L.latLng(stop.coordinates.lat, stop.coordinates.lng)
    );

    const routingControl = (L as any).Routing.control({
      waypoints: waypoints,
      routeWhileDragging: false,
      addWaypoints: false,
      createMarker: () => null,
      show: false,
      lineOptions: {
        styles: [{ color: "blue", opacity: 0.8, weight: 6 }],
      },
    }).addTo(mapInstance);

    return () => {
      mapInstance.removeControl(routingControl);
    };
  }, [mapInstance, routeData, L]);

  return null;
};

const MapComponent = ({ routeData }: { routeData: RouteData | null }) => {
  const { MapContainer, TileLayer, Marker, Popup } = (window as any)
    .ReactLeaflet;
  const L = (window as any).L;
  const [mapInstance, setMapInstance] = useState<any>(null);

  const createIcon = (color: string) => {
    return new L.Icon({
      iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-${color}.png`,
      shadowUrl:
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41],
    });
  };

  const icons = {
    default: createIcon("blue"),
    pickup: createIcon("green"),
    delivery: createIcon("red"),
  };

  const kochiPosition: [number, number] = [9.9312, 76.2673];
  const route = routeData?.data?.routes?.[0];

  return (
    <MapContainer
      whenCreated={setMapInstance}
      center={kochiPosition}
      zoom={13}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {!route && (
        <Marker position={kochiPosition} icon={icons.default}>
          <Popup>Kochi, Kerala, India.</Popup>
        </Marker>
      )}

      {route &&
        route.stops.map((stop) => (
          <Marker
            key={stop.sequence}
            position={[stop.coordinates.lat, stop.coordinates.lng]}
            icon={stop.type === "pickup" ? icons.pickup : icons.delivery}
          >
            <Popup>
              <b>
                Stop {stop.sequence}: {stop.type.toUpperCase()}
              </b>
              <br />
              Package: {stop.package_id}
              <br />
              Address: {stop.address}
              <br />
              ETA: {stop.eta}
            </Popup>
          </Marker>
        ))}

      {mapInstance && route && L && (
        <Routing mapInstance={mapInstance} routeData={routeData!} L={L} />
      )}
    </MapContainer>
  );
};

const GeographicalMap = ({ routeData }: { routeData: RouteData | null }) => {
  const [libsLoaded, setLibsLoaded] = useState(false);
  useEffect(() => {
    const L = (window as any).L;
    if (L && (window as any).ReactLeaflet && L.Routing) {
      setLibsLoaded(true);
      return;
    }

    const loadStyle = (href: string, id: string) => {
      if (!document.getElementById(id)) {
        const link = document.createElement("link");
        link.id = id;
        link.rel = "stylesheet";
        link.href = href;
        document.head.appendChild(link);
      }
    };

    const loadScript = (src: string, onLoad: () => void) => {
      const existingScript = document.querySelector(`script[src="${src}"]`);
      if (existingScript) {
        onLoad();
        return;
      }
      const script = document.createElement("script");
      script.src = src;
      script.onload = onLoad;
      script.onerror = () => console.error(`Failed to load: ${src}`);
      document.body.appendChild(script);
    };

    loadStyle(
      "https://unpkg.com/leaflet@1.7.1/dist/leaflet.css",
      "leaflet-css"
    );
    loadStyle(
      "https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css",
      "leaflet-routing-css"
    );

    loadScript("https://unpkg.com/leaflet@1.7.1/dist/leaflet.js", () => {
      loadScript(
        "https://unpkg.com/react-leaflet@3.2.5/umd/react-leaflet.min.js",
        () => {
          loadScript(
            "https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.js",
            () => {
              setLibsLoaded(true);
            }
          );
        }
      );
    });
  }, []);

  return (
    <div className="flex flex-col h-full p-4 md:p-6 bg-gray-100 dark:bg-gray-900 rounded-lg shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <MapIcon className="w-7 h-7 text-blue-500" />
        <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200">
          Geographical Map
        </h2>
      </div>
      <div className="flex-1 w-full h-full rounded-lg overflow-hidden border-2 border-gray-300 dark:border-gray-700">
        {libsLoaded ? (
          <MapComponent routeData={routeData} />
        ) : (
          <div className="w-full h-full bg-gray-200 dark:bg-gray-800 flex items-center justify-center">
            <p className="text-gray-500">Loading map...</p>
          </div>
        )}
      </div>
    </div>
  );
};

// --- MOCK API DATA ---
const mockApiResponse: RouteData = {
  status: "success",
  optimization_id: "OPT_20250913_001",
  timestamp: "2025-09-13T13:45:00Z",
  data: {
    routes: [
      {
        vehicle_id: "VAN001",
        driver_name: "John Smith",
        route_duration_minutes: 240,
        total_distance_km: 45.2,
        fuel_cost_usd: 23.5,
        stops: [
          {
            sequence: 1,
            type: "pickup",
            package_id: "PKG002",
            eta: "09:15",
            coordinates: { lat: 40.7128, lng: -74.006 },
            address: "123 Main St, New York, NY",
          },
          {
            sequence: 2,
            type: "delivery",
            package_id: "PKG002",
            eta: "09:45",
            coordinates: { lat: 40.7589, lng: -73.9851 },
            address: "456 Oak Ave, New York, NY",
          },
          {
            sequence: 3,
            type: "pickup",
            package_id: "PKG003",
            eta: "10:15",
            coordinates: { lat: 40.7295, lng: -73.9965 },
            address: "789 Broadway, New York, NY",
          },
        ],
      },
    ],
  },
};

// --- MAIN PAGE ---

export default function Page() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [routeData, setRouteData] = useState<RouteData | null>(null);

  // This function simulates fetching data from your backend
  const findRoute = () => {
    console.log("Simulating API call to find route...");
    // In a real app, you would use fetch() here:
    // fetch('/api/find-optimal-route')
    //   .then(res => res.json())
    //   .then(data => setRouteData(data));

    // For now, we use the mock data.
    setRouteData(mockApiResponse);
    alert("Route found and plotted on the map!");
  };

  return (
    <main className="h-screen w-full flex bg-gray-200 dark:bg-gray-950">
      <div
        className={`transition-all duration-300 ease-in-out ${
          isSidebarOpen ? "w-1/4 min-w-[350px] max-w-[450px]" : "w-0 min-w-0"
        }`}
      >
        <div className="h-full p-2 md:p-4 pr-0 overflow-hidden">
          <AssistantPanel onFindRoute={findRoute} />
        </div>
      </div>

      <div className="relative flex-shrink-0">
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="z-20 p-1 bg-gray-700 text-white rounded-full hover:bg-gray-600 absolute top-1/2 -translate-y-1/2 transition-all"
          style={{ left: isSidebarOpen ? "-12px" : "12px" }}
        >
          {isSidebarOpen ? (
            <ChevronLeftIcon className="w-5 h-5" />
          ) : (
            <ChevronRightIcon className="w-5 h-5" />
          )}
        </button>
      </div>

      <div className="flex-1">
        <div className="h-full p-2 md:p-4">
          <GeographicalMap routeData={routeData} />
        </div>
      </div>
    </main>
  );
}
