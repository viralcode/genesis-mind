import logging
import threading
from typing import Dict, Any, List
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy as np
import torch
from datetime import datetime

logger = logging.getLogger("genesis.dashboard.server")

# Global reference to the mind
_mind_instance = None


class DashboardServer:
    """
    A lightweight Flask server that exposes the Genesis Mind's
    internal state as a JSON API, and serves the web dashboard UI.
    Runs in a background thread.
    """
    def __init__(self, mind, port: int = 5000):
        global _mind_instance
        _mind_instance = mind
        
        self.port = port
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        CORS(self.app)
        
        # Setup routes
        self._setup_routes()
        
        self._thread = None
        self._running = False
        
        # Suppress noisy werkzeug logs
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template("index.html")
            
        @self.app.route('/api/state')
        def state():
            if not _mind_instance:
                return jsonify({"status": "booting..."})
            return jsonify(self._build_state_payload(_mind_instance))

    def _tensor_to_list(self, tensor: torch.Tensor, max_items: int = 50) -> List[float]:
        """Safely convert a tensor to a flat list for visualization."""
        if tensor is None:
            return []
        try:
            # Flatten, convert to float, take first max_items
            flat = tensor.detach().cpu().flatten().numpy()
            return [float(x) for x in flat[:max_items]]
        except Exception:
            return []

    def _build_state_payload(self, mind) -> Dict[str, Any]:
        """
        Extract the ENTIRE biological and neural state of the system.
        """
        state = {}
        
        try:
            # 1. Core Identity & Phase
            state["core"] = {
                "phase": mind.development.current_phase,
                "phase_name": mind.development.current_phase_name,
                "age_seconds": (datetime.now() - datetime.fromisoformat(mind.axioms.birth_time)).total_seconds() if hasattr(mind, 'axioms') else 0,
                "grammar_mode": mind.grammar.mode,
            }
            
            # 2. Neurochemistry (4 functional chemicals)
            state["neurochemistry"] = {
                "dopamine": float(mind.neurochemistry.dopamine.level),
                "cortisol": float(mind.neurochemistry.cortisol.level),
                "serotonin": float(mind.neurochemistry.serotonin.level),
                "oxytocin": float(mind.neurochemistry.oxytocin.level)
            }
            
            # 3. Drive System (8 Maslow Tiers)
            state["drives"] = mind.drives.get_status()
            
            # 4. Emotional State (8-Dim Vector)
            state["emotions"] = mind.emotional_state.get_vector().tolist()
            
            # 5. Attention & Working Memory
            active_slots = [
                {"concept": item.key, "salience": item.salience, "source": "Mental Buffer"}
                for item in mind.working_memory.get_active_items()
            ]
            state["working_memory"] = {
                "capacity": mind.working_memory.capacity,
                "usage": len(active_slots),
                "slots": active_slots
            }
            state["attention_threshold"] = mind.attention.salience_threshold
            
            # 6. Brain Daemon Threads
            threads = {}
            if hasattr(mind, "_brain") and mind._brain:
                for name, t in mind._brain._threads.items():
                    threads[name] = {"ticks": t._tick_count, "errors": t._errors}
            state["threads"] = threads
            
            # 7. Senses (Latest captures)
            state["senses"] = {
                "vision": "Camera Active (3s cycle)" if mind._eyes else "Sleeping",
                "auditory": "Mic Active (3s chunks)",
                "proprioception": mind.proprioception.get_status()
            }
            
            # 8. Neural Subconscious (The 4 Layers)
            hidden_state_viz = []
            if hasattr(mind.subconscious.personality, "_hidden_state") and mind.subconscious.personality._hidden_state is not None:
                hidden_state_viz = self._tensor_to_list(mind.subconscious.personality._hidden_state, max_items=128)
                
            state["neural"] = {
                "total_parameters": 593445, # mind.subconscious.get_total_parameters() if available
                "layer1_limbic": {
                    "params": 59620,
                    "last_dopamine": float(mind.neurochemistry.dopamine.level)
                },
                "layer2_binding": {
                    "params": 131457,
                    "learned_concepts": mind.semantic_memory.count()
                },
                "layer3_personality": {
                    "params": 311296,
                    "hidden_state_activation": hidden_state_viz,
                    "total_experiences": mind.subconscious.personality._total_experiences
                },
                "layer4_world_model": {
                    "params": 91072,
                    "total_predictions": mind.subconscious.world_model._predictions_made if hasattr(mind.subconscious, 'world_model') else 0,
                    "last_loss": float(mind.subconscious.world_model._total_loss / max(1, mind.subconscious.world_model._predictions_made)) if hasattr(mind.subconscious, 'world_model') else 0.0
                },
                "meta_controller": {
                    "params": 61957
                }
            }
            
            # 9. Neural Network Graph (Semantic/Binding visual map)
            nodes = []
            edges = []
            if hasattr(mind, 'semantic_memory') and mind.semantic_memory:
                try:
                    concepts = mind.semantic_memory.get_all_concepts()
                    for c in concepts:
                        nodes.append({
                            "id": c.word, 
                            "label": c.word, 
                            "group": "concept", 
                            "value": getattr(c, 'strength', 1.0)
                        })
                        for rel in getattr(c, 'relationships', []):
                            edges.append({"from": c.word, "to": rel})
                except Exception as e:
                    logger.error("Failed to build semantic map: %s", e)
            
            state["network_graph"] = {"nodes": nodes, "edges": edges}
            
            # 10. Activity Stream
            stream = []
            if hasattr(mind, '_activity_stream'):
                stream = mind._activity_stream[-30:]  # Last 30 events
            state["stream"] = stream
            
            # 11. Memory Storage
            state["memory"] = {
                "semantic_concepts": mind.semantic_memory.count(),
                "episodic_count": mind.episodic_memory.count(),
                "replay_buffer": len(mind.subconscious.replay_buffer) if hasattr(mind.subconscious, 'replay_buffer') else 0
            }
            
            # 10. Language & ToM
            state["cognition"] = {
                "theory_of_mind": mind.theory_of_mind.is_active,
                "known_words": list(mind.grammar.ngram_model.keys())[:10] if hasattr(mind.grammar, 'ngram_model') else []
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error building state payload: {e}")
            state["error"] = str(e)
            
        return state

    def start(self):
        """Start the Flask server in a background daemon thread."""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting Web Dashboard API on port {self.port}")
        
        self._thread = threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False),
            name="genesis-dashboard",
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the dashboard."""
        self._running = False
