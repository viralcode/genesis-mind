import logging
import threading
from typing import Dict, Any, List
from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
import numpy as np
import torch
import time
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
            
        @self.app.route('/network')
        def network():
            return render_template("network.html")

        @self.app.route('/api/camera')
        def camera_feed():
            """MJPEG stream of what Genesis sees through its eyes."""
            return Response(
                self._generate_camera_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/api/neural_activations')
        def neural_activations():
            """Real-time neural layer activations for visualization."""
            if not _mind_instance:
                return jsonify({})
            return jsonify(self._build_neural_activations(_mind_instance))
            
        @self.app.route('/api/debug_camera')
        def debug_camera():
            """Return stats about the current camera frame."""
            if not _mind_instance or not _mind_instance._eyes:
                return jsonify({"error": "No eyes"})
            frame = getattr(_mind_instance._eyes, '_last_frame_full', None)
            if frame is None:
                return jsonify({"error": "No frame"})
            return jsonify({
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "min": float(frame.min()),
                "max": float(frame.max()),
                "mean": float(frame.mean())
            })

        @self.app.route('/api/state')
        def state():
            if not _mind_instance:
                return jsonify({"status": "booting..."})
            return jsonify(self._build_state_payload(_mind_instance))

    def _generate_camera_frames(self):
        """Yield MJPEG frames from the cached last frame (no competing camera reads)."""
        import time
        import cv2
        while True:
            try:
                if _mind_instance and _mind_instance._eyes:
                    eyes = _mind_instance._eyes
                    # Prefer full-res frame for dashboard, fallback to 64x64 cortex input
                    frame = getattr(eyes, '_last_frame_full', None)
                    if frame is None:
                        frame = getattr(eyes, '_last_frame', None)
                    if frame is not None:
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    else:
                        # Yield a blank placeholder frame to keep stream alive
                        blank = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(blank, "Awaiting vision daemon...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        _, jpeg = cv2.imencode('.jpg', blank)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                logger.error("Camera stream error: %s", e)
            time.sleep(0.5)  # ~2 FPS to match vision thread

    def _build_neural_activations(self, mind) -> dict:
        """Extract real-time neural activations for heatmap visualization."""
        activations = {}
        try:
            # Personality GRU hidden state
            if hasattr(mind.subconscious.personality, '_hidden_state') and mind.subconscious.personality._hidden_state is not None:
                activations['personality_hidden'] = self._tensor_to_list(
                    mind.subconscious.personality._hidden_state, max_items=128
                )

            # Limbic system — last reaction values
            activations['limbic'] = {
                'dopamine': float(mind.neurochemistry.dopamine.level),
                'cortisol': float(mind.neurochemistry.cortisol.level),
                'serotonin': float(mind.neurochemistry.serotonin.level),
                'oxytocin': float(mind.neurochemistry.oxytocin.level),
            }

            # Meta-controller routing weights
            mc = mind.subconscious.meta_controller
            if hasattr(mc, '_avg_weights') and hasattr(mc, 'MODULE_NAMES'):
                activations['routing'] = {
                    name: round(float(mc._avg_weights[i]), 3)
                    for i, name in enumerate(mc.MODULE_NAMES)
                    if i < len(mc._avg_weights)
                }

            # VQ codebook usage heatmap
            if hasattr(mind, 'sensorimotor') and mind.sensorimotor:
                # In V7, the codebook is within auditory_cortex (AcousticWordMemory)
                if hasattr(mind.sensorimotor, 'auditory_cortex') and hasattr(mind.sensorimotor.auditory_cortex, 'codebook'):
                    vq = mind.sensorimotor.auditory_cortex.codebook
                    if hasattr(vq, '_usage'):
                        usage = vq._usage.tolist()
                        activations['vq_usage'] = usage[:256]

            # Growth stats
            activations['total_params'] = mind.subconscious.get_total_params()
            if hasattr(mind, 'sensorimotor') and mind.sensorimotor:
                activations['total_params'] += sum(
                    p.numel() for p in mind.sensorimotor.auditory_cortex.encoder.parameters()
                ) + sum(
                    p.numel() for p in mind.sensorimotor.acoustic_brain.model.parameters()
                )

            # Sensory buffers (co-occurrence state)
            if hasattr(mind, '_brain') and mind._brain:
                bd = mind._brain
                now = time.time()
                
                v_age = now - getattr(bd, '_recent_visual_time', 0)
                v_fresh = getattr(bd, '_recent_visual_embedding', None) is not None and v_age < 3.0
                
                a_age = now - getattr(bd, '_recent_heard_time', 0)
                a_fresh = getattr(bd, '_recent_heard_words', []) if a_age < 3.0 else []
                
                activations['sensory_buffers'] = {
                    'visual_fresh': v_fresh,
                    'visual_age': round(v_age, 1) if getattr(bd, '_recent_visual_time', 0) > 0 else -1,
                    'heard_words': a_fresh[:5],
                    'heard_age': round(a_age, 1) if getattr(bd, '_recent_heard_time', 0) > 0 else -1,
                }

        except Exception as e:
            activations['error'] = str(e)
        return activations

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
                "is_sleeping": getattr(mind.sleep_cycle, 'is_sleeping', False) if hasattr(mind, 'sleep_cycle') else False,
                "current_sleep_phase": getattr(mind.sleep_cycle, 'current_phase_name', 'awake') if hasattr(mind, 'sleep_cycle') else 'awake',
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
            state["mood_baseline"] = mind.emotional_state.get_mood().tolist()
            
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
                        # Filter out malformed entries (ChromaDB artifacts with paths, long IDs, etc.)
                        word = c.word
                        if not word or '/' in word or len(word) > 30:
                            continue
                        strength = getattr(c, 'strength', 1.0)
                        nodes.append({
                            "id": word, 
                            "label": word, 
                            "group": "concept", 
                            "title": f"Strength: {strength:.2f}",
                            "strength": round(strength, 3)
                        })
                        for rel in getattr(c, 'relationships', []):
                            if rel and '/' not in rel and len(rel) <= 30:
                                edges.append({"from": word, "to": rel})
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
            
            # 10. Language Acquisition (V6)
            language_data = {
                "grammar_mode": "unknown",
                "babbling": {},
                "joint_attention": {},
                "ngram": {},
            }
            try:
                language_data["grammar_mode"] = mind.grammar.mode
                if hasattr(mind, 'babbling_engine'):
                    language_data["babbling"] = mind.babbling_engine.get_status()
                if hasattr(mind, 'joint_attention'):
                    language_data["joint_attention"] = mind.joint_attention.get_status()
                language_data["ngram"] = mind.grammar.get_ngram_stats()
            except Exception as e:
                logger.error("Failed to build language data: %s", e)
            state["language_acquisition"] = language_data

            # 11. Theory of Mind
            state["cognition"] = {
                "theory_of_mind": mind.theory_of_mind.is_active,
            }

            # 12. Acoustic Neural Pipeline (V7)
            if hasattr(mind, 'sensorimotor'):
                try:
                    state["acoustic_pipeline"] = mind.sensorimotor.get_stats()
                except Exception as e:
                    logger.error("Failed to get acoustic stats: %s", e)
                    state["acoustic_pipeline"] = {}

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
