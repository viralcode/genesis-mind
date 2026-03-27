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

        @self.app.route('/explorer')
        def explorer():
            return render_template("explorer.html")

        @self.app.route('/api/explorer_summary')
        def explorer_summary():
            """All networks' stats in one payload for the overview grid."""
            if not _mind_instance:
                return jsonify({"error": "Mind not ready"})
            return jsonify(self._build_explorer_summary(_mind_instance))

        @self.app.route('/api/network_deep/<network_name>')
        def network_deep(network_name):
            """Deep introspection of a single network: architecture, weights, gradients."""
            if not _mind_instance:
                return jsonify({"error": "Mind not ready"})
            return jsonify(self._build_network_deep(_mind_instance, network_name))

        @self.app.route('/mind')
        def mind_page():
            return render_template("mind.html")

        @self.app.route('/api/mind_state')
        def mind_state():
            """What's actually inside the mind — concepts, memories, thoughts, emotions."""
            if not _mind_instance:
                return jsonify({"error": "Mind not ready"})
            return jsonify(self._build_mind_state(_mind_instance))

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

        # ═══ Diagnostic API Endpoints (Observability Pillar) ═══

        @self.app.route('/api/training_history')
        def training_history():
            """Per-network loss history for dashboard loss curve visualization."""
            if not _mind_instance:
                return jsonify({})
            try:
                history = _mind_instance.subconscious.get_training_history()
                # Convert to JSON-friendly format
                result = {}
                for name, entries in history.items():
                    result[name] = {
                        "timestamps": [t for t, _ in entries],
                        "losses": [l for _, l in entries],
                        "count": len(entries),
                    }
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)})

        @self.app.route('/api/embedding_map')
        def embedding_map():
            """2D t-SNE projection of all concept embeddings."""
            if not _mind_instance:
                return jsonify({})
            try:
                return jsonify(self._build_embedding_map(_mind_instance))
            except Exception as e:
                return jsonify({"error": str(e)})

        @self.app.route('/api/binding_accuracy')
        def binding_accuracy():
            """Binding consistency: how consistently same concepts produce similar embeddings."""
            if not _mind_instance:
                return jsonify({})
            try:
                return jsonify(self._compute_binding_accuracy(_mind_instance))
            except Exception as e:
                return jsonify({"error": str(e)})

        @self.app.route('/api/replay_stats')
        def replay_stats():
            """Histogram of surprise/emotion/drive values in the replay buffer."""
            if not _mind_instance:
                return jsonify({})
            try:
                return jsonify(self._build_replay_stats(_mind_instance))
            except Exception as e:
                return jsonify({"error": str(e)})

        @self.app.route('/api/session_log')
        def session_log():
            """Last N interaction events with internal state."""
            if not _mind_instance:
                return jsonify([])
            try:
                log = getattr(_mind_instance, '_interaction_log', [])
                return jsonify(list(log)[-200:])
            except Exception as e:
                return jsonify({"error": str(e)})

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
            try:
                activations['total_params'] = mind.subconscious.get_total_params()
                if hasattr(mind, 'sensorimotor') and mind.sensorimotor:
                    activations['total_params'] += sum(
                        p.numel() for p in mind.sensorimotor.auditory_cortex.encoder.parameters()
                    ) + sum(
                        p.numel() for p in mind.sensorimotor.acoustic_brain.model.parameters()
                    )
            except Exception:
                activations['total_params'] = -1  # Param counting failed (torch.compile wrapper)

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
                    'audio_active': getattr(bd, '_recent_audio_active', False) and (now - getattr(bd, '_recent_audio_time', 0)) < 5.0,
                    'audio_age': round(now - getattr(bd, '_recent_audio_time', 0), 1) if getattr(bd, '_recent_audio_time', 0) > 0 else -1,
                    'co_occurrence_active': getattr(bd, '_co_occurrence_active', False) and (now - getattr(bd, '_co_occurrence_time', 0)) < 5.0,
                }
                
                # Visual Saliency signals (from VisualStimulusAnalyzer)
                saliency = getattr(bd, '_last_saliency', {})
                if saliency:
                    activations['visual_saliency'] = {
                        'motion': round(saliency.get('motion', 0), 4),
                        'novelty': round(saliency.get('novelty', 0), 4),
                        'complexity': round(saliency.get('complexity', 0), 4),
                        'luminance_change': round(saliency.get('luminance_change', 0), 4),
                        'overall_saliency': round(saliency.get('overall_saliency', 0), 4),
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
            
            # 6. Brain Daemon Threads (with profiling)
            threads = {}
            if hasattr(mind, "_brain") and mind._brain:
                threads = mind._brain.get_profiling()
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
            
            # 10. Language Acquisition (V6 -> V7)
            language_data = {
                "grammar_mode": "unknown",
                "babbling": {},
                "joint_attention": {},
                "ngram": {},
                "acoustic_memory": {}
            }
            try:
                language_data["grammar_mode"] = mind.grammar.mode
                if hasattr(mind, 'babbling_engine'):
                    language_data["babbling"] = mind.babbling_engine.get_status()
                if hasattr(mind, 'joint_attention'):
                    language_data["joint_attention"] = mind.joint_attention.get_status()
                language_data["ngram"] = mind.grammar.get_ngram_stats()
                
                # V7: Pure acoustic vocabulary
                if hasattr(mind, 'acoustic_word_memory'):
                    language_data["acoustic_memory"] = mind.acoustic_word_memory.get_stats()
                    
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

    # ═══════════════════════════════════════════════════════════════════
    # Neural Explorer — Deep Introspection API
    # ═══════════════════════════════════════════════════════════════════

    def _get_network_registry(self, mind) -> dict:
        """Map of network_name → (module, category, description)."""
        registry = {}
        sub = mind.subconscious
        
        # Subconscious core networks
        registry["limbic"] = (sub.limbic_system, "emotional", "Emotional reactions → neurochemistry")
        registry["binding"] = (sub.binding_network, "cognitive", "Multi-modal concept binding")
        registry["personality"] = (sub.personality, "cognitive", "GRU stream of consciousness")
        registry["world_model"] = (sub.world_model, "cognitive", "Next-state prediction")
        registry["meta_controller"] = (sub.meta_controller, "cognitive", "Attention routing across modules")
        
        # Visual cortex (lives on the Eyes sensor)
        if hasattr(mind, '_eyes') and mind._eyes and hasattr(mind._eyes, '_visual_cortex') and mind._eyes._visual_cortex:
            registry["visual_cortex"] = (mind._eyes._visual_cortex, "sensory", "Conv autoencoder (pixels → 64-dim)")
        
        # Acoustic pipeline
        if hasattr(mind, 'sensorimotor') and mind.sensorimotor:
            sm = mind.sensorimotor
            if hasattr(sm, 'auditory_cortex'):
                registry["auditory_cortex"] = (sm.auditory_cortex, "acoustic", "Mel spectrogram → embeddings")
            if hasattr(sm, 'acoustic_brain'):
                registry["acoustic_lm"] = (sm.acoustic_brain, "acoustic", "Transformer language model")
            if hasattr(sm, 'vocoder'):
                registry["vocoder"] = (sm.vocoder, "acoustic", "Neural vocoder → speech")
            if hasattr(sm, 'vq_codebook'):
                registry["vq_codebook"] = (sm.vq_codebook, "acoustic", "VQ phoneme discovery (256 codes)")
        
        return registry

    def _build_explorer_summary(self, mind) -> dict:
        """All networks' stats for the overview grid."""
        summary = {"networks": {}, "system": {}, "curriculum": {}}
        
        try:
            registry = self._get_network_registry(mind)
            
            for name, (module, category, description) in registry.items():
                try:
                    stats = module.get_stats() if hasattr(module, 'get_stats') else {}
                    
                    # Compute health indicator
                    avg_loss = stats.get("avg_loss", stats.get("avg_surprise", -1))
                    training_steps = stats.get("training_steps", stats.get("train_steps", 0))
                    params = stats.get("params", stats.get("total_params", 0))
                    
                    if avg_loss < 0:
                        health = "idle"
                    elif avg_loss < 0.1:
                        health = "good"
                    elif avg_loss < 0.5:
                        health = "learning"
                    else:
                        health = "early"
                    
                    summary["networks"][name] = {
                        "category": category,
                        "description": description,
                        "params": params,
                        "training_steps": training_steps,
                        "avg_loss": round(float(avg_loss), 6) if avg_loss >= 0 else None,
                        "health": health,
                        **{k: v for k, v in stats.items() 
                           if k not in ("params", "total_params", "training_steps", "train_steps", "avg_loss", "avg_surprise")},
                    }
                except Exception as e:
                    summary["networks"][name] = {"error": str(e), "category": category, "description": description}
            
            # Curriculum gates
            sub = mind.subconscious
            summary["curriculum"] = {
                "visual_cortex_loss": round(getattr(sub, '_visual_cortex_loss', 1.0), 6),
                "binding_gate": getattr(sub, '_binding_gate_open', False),
                "personality_gate": getattr(sub, '_personality_gate_open', False),
                "world_model_gate": getattr(sub, '_world_model_gate_open', False),
            }
            
            # System-wide stats
            summary["system"] = {
                "total_params": sum(
                    s.get("params", 0) for s in summary["networks"].values() if isinstance(s.get("params"), (int, float))
                ),
                "total_training_steps": sum(
                    s.get("training_steps", 0) for s in summary["networks"].values() if isinstance(s.get("training_steps"), (int, float))
                ),
                "growth_events": mind.neuroplasticity._total_growth_events if hasattr(mind, 'neuroplasticity') else 0,
                "phase": mind.development.current_phase if hasattr(mind, 'development') else 0,
                "phase_name": mind.development.current_phase_name if hasattr(mind, 'development') else "Unknown",
            }
        except Exception as e:
            summary["error"] = str(e)
        
        return summary

    def _build_network_deep(self, mind, network_name: str) -> dict:
        """Deep introspection of a single network."""
        result = {"name": network_name, "layers": [], "weight_stats": {}, "activations": {}}
        
        try:
            registry = self._get_network_registry(mind)
            if network_name not in registry:
                return {"error": f"Unknown network: {network_name}", "available": list(registry.keys())}
            
            module, category, description = registry[network_name]
            result["category"] = category
            result["description"] = description
            
            # Get stats
            if hasattr(module, 'get_stats'):
                result["stats"] = module.get_stats()
            
            # Find the actual nn.Module to inspect
            net = None
            if hasattr(module, 'network') and isinstance(module.network, torch.nn.Module):
                net = module.network
            elif hasattr(module, 'encoder') and isinstance(module.encoder, torch.nn.Module):
                # Visual cortex — inspect encoder
                net = module.encoder
                result["has_decoder"] = True
            elif hasattr(module, 'model') and isinstance(module.model, torch.nn.Module):
                net = module.model
            elif isinstance(module, torch.nn.Module):
                net = module
            
            if net is not None:
                # Layer architecture
                layers = []
                for name, param in net.named_parameters():
                    layers.append({
                        "name": name,
                        "shape": list(param.shape),
                        "params": param.numel(),
                        "requires_grad": param.requires_grad,
                        "dtype": str(param.dtype),
                    })
                result["layers"] = layers
                
                # Weight distributions per named module
                weight_stats = {}
                for name, param in net.named_parameters():
                    try:
                        data = param.detach().cpu().float()
                        # Compute histogram (20 bins)
                        hist_values = data.flatten().numpy()
                        counts, bin_edges = np.histogram(hist_values, bins=20)
                        
                        weight_stats[name] = {
                            "mean": round(float(data.mean()), 6),
                            "std": round(float(data.std()), 6) if data.numel() > 1 else 0.0,
                            "min": round(float(data.min()), 6),
                            "max": round(float(data.max()), 6),
                            "norm": round(float(data.norm()), 4),
                            "histogram": {
                                "counts": counts.tolist(),
                                "edges": [round(float(e), 4) for e in bin_edges.tolist()],
                            },
                        }
                        # Gradient info
                        if param.grad is not None:
                            weight_stats[name]["grad_norm"] = round(float(param.grad.detach().cpu().norm()), 6)
                    except Exception:
                        pass
                result["weight_stats"] = weight_stats
            
            # Network-specific activations
            if network_name == "personality":
                if hasattr(module, '_hidden_state') and module._hidden_state is not None:
                    result["activations"]["hidden_state"] = self._tensor_to_list(module._hidden_state, max_items=256)
                result["activations"]["experience_buffer_size"] = len(getattr(module, '_experience_buffer', []))
            
            elif network_name == "meta_controller":
                if hasattr(module, '_avg_weights') and hasattr(module, 'MODULE_NAMES'):
                    result["activations"]["routing"] = {
                        name: round(float(module._avg_weights[i]), 4)
                        for i, name in enumerate(module.MODULE_NAMES)
                        if i < len(module._avg_weights)
                    }
            
            elif network_name == "vq_codebook":
                if hasattr(module, '_usage'):
                    usage = module._usage.tolist()
                    result["activations"]["codebook_usage"] = usage[:256]
                    result["activations"]["active_codes"] = sum(1 for u in usage if u > 0)
                    result["activations"]["total_codes"] = len(usage)
            
            elif network_name == "visual_cortex":
                result["activations"]["frames_seen"] = getattr(module, '_frames_seen', 0)
                result["activations"]["train_steps"] = getattr(module, '_train_steps', 0)
                # Also get decoder layer info
                if hasattr(module, 'decoder') and isinstance(module.decoder, torch.nn.Module):
                    decoder_layers = []
                    for name, param in module.decoder.named_parameters():
                        decoder_layers.append({
                            "name": f"decoder.{name}",
                            "shape": list(param.shape),
                            "params": param.numel(),
                        })
                    result["decoder_layers"] = decoder_layers
        
        except Exception as e:
            result["error"] = str(e)
        
        return result

    # ═══════════════════════════════════════════════════════════════════
    # Diagnostic Helpers (Observability Pillar)
    # ═══════════════════════════════════════════════════════════════════

    def _build_embedding_map(self, mind) -> dict:
        """Compute 2D t-SNE projection of all concept embeddings."""
        result = {"points": [], "cached": False}
        
        # Check cache
        now = time.time()
        if hasattr(self, '_embedding_cache') and self._embedding_cache:
            cache_age = now - self._embedding_cache.get('timestamp', 0)
            if cache_age < 30.0:  # 30s cache
                return self._embedding_cache['data']
        
        concepts = mind.semantic_memory.get_all_concepts()
        embeddings = []
        labels = []
        strengths = []
        
        for c in concepts:
            if c.text_embedding and len(c.text_embedding) > 0:
                word = c.word
                if not word or '/' in word or len(word) > 30:
                    continue
                embeddings.append(c.text_embedding)
                labels.append(word)
                strengths.append(getattr(c, 'strength', 1.0))
        
        if len(embeddings) < 3:
            result["error"] = "Need at least 3 concepts for t-SNE"
            return result
        
        try:
            from sklearn.manifold import TSNE
            X = np.array(embeddings, dtype=np.float32)
            perplexity = min(30, max(2, len(embeddings) - 1))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=300)
            coords = tsne.fit_transform(X)
            
            for i, (label, strength) in enumerate(zip(labels, strengths)):
                result["points"].append({
                    "label": label,
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                    "strength": round(strength, 3),
                })
            
            # Cache result
            self._embedding_cache = {"timestamp": now, "data": result}
            
        except ImportError:
            result["error"] = "sklearn not available for t-SNE"
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def _compute_binding_accuracy(self, mind) -> dict:
        """
        Binding consistency: for each concept with visual+text pairs,
        how consistently does the binder produce similar outputs?
        
        Runs each concept through the binding network twice and
        measures cosine similarity of the outputs.
        """
        result = {"per_concept": {}, "avg_consistency": 0.0, "total_concepts": 0}
        
        concepts = mind.semantic_memory.get_all_concepts()
        consistencies = []
        
        for c in concepts:
            if c.visual_embedding is None or c.text_embedding is None:
                continue
            word = c.word
            if not word or '/' in word or len(word) > 30:
                continue
            
            try:
                v = np.array(c.visual_embedding, dtype=np.float32)
                a = np.array(c.text_embedding, dtype=np.float32)
                
                # Bind twice (results should be deterministic but we're checking)
                binding1 = mind.subconscious.binding_network.bind(v, a)
                binding2 = mind.subconscious.binding_network.bind(v, a)
                
                # Cosine similarity
                dot = np.dot(binding1, binding2)
                norm = np.linalg.norm(binding1) * np.linalg.norm(binding2)
                consistency = float(dot / (norm + 1e-8))
                
                consistencies.append(consistency)
                result["per_concept"][word] = round(consistency, 4)
            except Exception:
                pass
        
        result["total_concepts"] = len(consistencies)
        result["avg_consistency"] = round(float(np.mean(consistencies)) if consistencies else 0.0, 4)
        
        return result

    def _build_replay_stats(self, mind) -> dict:
        """Histogram of surprise/emotion/drive values in the replay buffer."""
        result = {"buffer_size": 0, "histograms": {}}
        
        buffer = list(mind.subconscious.replay_buffer)
        result["buffer_size"] = len(buffer)
        
        if not buffer:
            return result
        
        for key in ['surprise', 'emotional_intensity', 'drive_hunger']:
            values = [exp.get(key, 0) for exp in buffer]
            if values:
                arr = np.array(values)
                counts, edges = np.histogram(arr, bins=15)
                result["histograms"][key] = {
                    "counts": counts.tolist(),
                    "edges": [round(float(e), 4) for e in edges.tolist()],
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                    "max": round(float(arr.max()), 4),
                }
        
        return result

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

    # =========================================================================
    # Mind State Builder — what's actually inside the mind
    # =========================================================================

    def _build_mind_state(self, mind):
        """Build a comprehensive view of what's forming inside the mind."""
        import time as _time
        state = {
            "timestamp": _time.time(),
        }

        # 1. Concepts learned (semantic memory)
        try:
            concepts = []
            if hasattr(mind, 'semantic_memory'):
                sm = mind.semantic_memory
                for name, concept in getattr(sm, '_concepts', {}).items():
                    c = {"name": name}
                    if hasattr(concept, 'strength'):
                        c["strength"] = round(concept.strength, 3)
                    if hasattr(concept, 'embedding') and concept.embedding is not None:
                        c["has_embedding"] = True
                        c["embedding_dim"] = len(concept.embedding) if hasattr(concept.embedding, '__len__') else 0
                    else:
                        c["has_embedding"] = False
                    if hasattr(concept, 'associations'):
                        c["associations"] = list(concept.associations.keys())[:5] if isinstance(concept.associations, dict) else []
                    if hasattr(concept, 'timestamp'):
                        c["learned_at"] = str(concept.timestamp)
                    if hasattr(concept, 'modality'):
                        c["modality"] = concept.modality
                    concepts.append(c)
                concepts.sort(key=lambda x: x.get("strength", 0), reverse=True)
            state["concepts"] = concepts
            state["concept_count"] = len(concepts)
        except Exception as e:
            state["concepts"] = []
            state["concept_count"] = 0

        # 2. Working memory — what's currently "in mind"
        try:
            wm_items = []
            if hasattr(mind, 'working_memory'):
                wm = mind.working_memory
                if hasattr(wm, '_items'):
                    for key, item in wm._items.items():
                        entry = {"key": str(key)}
                        if hasattr(item, 'content'):
                            entry["content"] = str(item.content)[:100]
                        if hasattr(item, 'salience'):
                            entry["salience"] = round(float(item.salience), 3)
                        if hasattr(item, 'timestamp'):
                            entry["age_sec"] = round(_time.time() - item.timestamp, 1)
                        wm_items.append(entry)
                    wm_items.sort(key=lambda x: x.get("salience", 0), reverse=True)
            state["working_memory"] = wm_items
        except Exception:
            state["working_memory"] = []

        # 3. Emotional state
        try:
            emotions = {}
            if hasattr(mind, 'emotional_state'):
                es = mind.emotional_state
                if hasattr(es, 'get_state'):
                    emotions = es.get_state()
                elif hasattr(es, '_dimensions'):
                    emotions = {dim: round(val, 3) for dim, val in es._dimensions.items()}
            state["emotions"] = emotions
        except Exception:
            state["emotions"] = {}

        # 4. Neurochemistry
        try:
            chemicals = {}
            if hasattr(mind, 'neurochemistry'):
                nc = mind.neurochemistry
                for name in ['dopamine', 'cortisol', 'serotonin', 'oxytocin']:
                    chem = getattr(nc, name, None)
                    if chem and hasattr(chem, 'level'):
                        chemicals[name] = round(float(chem.level), 4)
            state["neurochemistry"] = chemicals
        except Exception:
            state["neurochemistry"] = {}

        # 5. Drives — what Genesis wants
        try:
            drives = {}
            if hasattr(mind, 'drives'):
                ds = mind.drives
                for attr_name in dir(ds):
                    drive = getattr(ds, attr_name, None)
                    if hasattr(drive, 'level') and hasattr(drive, 'name'):
                        drives[drive.name] = {
                            "level": round(float(drive.level), 3),
                            "satisfied": float(drive.level) < 0.3,
                        }
            state["drives"] = drives
        except Exception:
            state["drives"] = {}

        # 6. Episodic memory — recent experiences
        try:
            episodes = []
            if hasattr(mind, 'episodic_memory'):
                em = mind.episodic_memory
                recent = getattr(em, '_episodes', [])
                for ep in recent[-20:]:  # Last 20
                    entry = {}
                    if hasattr(ep, 'summary'):
                        entry["summary"] = str(ep.summary)[:100]
                    if hasattr(ep, 'type'):
                        entry["type"] = str(ep.type)
                    if hasattr(ep, 'timestamp'):
                        entry["timestamp"] = str(ep.timestamp)
                    if hasattr(ep, 'emotional_valence'):
                        entry["emotion"] = round(float(ep.emotional_valence), 2)
                    episodes.append(entry)
                episodes.reverse()  # Newest first
            state["episodes"] = episodes
            state["episode_count"] = len(getattr(mind.episodic_memory, '_episodes', []))
        except Exception:
            state["episodes"] = []
            state["episode_count"] = 0

        # 7. Vocabulary — acoustic words learned
        try:
            vocab = {}
            if hasattr(mind, 'acoustic_word_memory'):
                awm = mind.acoustic_word_memory
                stats = awm.get_stats()
                vocab["words"] = stats.get("vocabulary", [])
                vocab["total_exemplars"] = stats.get("total_exemplars", 0)
                vocab["recognition_rate"] = round(stats.get("recognition_rate", 0), 3)
                vocab["total_recognitions"] = stats.get("total_recognitions", 0)
            state["vocabulary"] = vocab
        except Exception:
            state["vocabulary"] = {}

        # 8. Grammar — what language patterns have been heard
        try:
            grammar = {}
            if hasattr(mind, 'grammar'):
                g = mind.grammar
                if hasattr(g, 'get_stats'):
                    gs = g.get_stats()
                    grammar["words_heard"] = gs.get("words_heard", 0)
                    grammar["sentences_heard"] = gs.get("sentences_heard", 0)
                    grammar["mode"] = gs.get("mode", "unknown")
                    # Top words by frequency
                    if hasattr(g, '_ngram') and hasattr(g._ngram, 'word_counts'):
                        sorted_words = sorted(
                            g._ngram.word_counts.items(),
                            key=lambda x: x[1], reverse=True
                        )[:20]
                        grammar["top_words"] = [
                            {"word": w, "count": c} for w, c in sorted_words
                        ]
            state["grammar"] = grammar
        except Exception:
            state["grammar"] = {}

        # 9. Cross-modal bindings
        try:
            bindings = []
            if hasattr(mind, 'joint_attention'):
                ja = mind.joint_attention
                if hasattr(ja, '_bindings'):
                    for b in ja._bindings[-20:]:
                        entry = {}
                        if hasattr(b, 'visual_concept'):
                            entry["visual"] = str(b.visual_concept)
                        if hasattr(b, 'auditory_word'):
                            entry["auditory"] = str(b.auditory_word)
                        if hasattr(b, 'strength'):
                            entry["strength"] = round(float(b.strength), 3)
                        bindings.append(entry)
            state["cross_modal_bindings"] = bindings
        except Exception:
            state["cross_modal_bindings"] = []

        # 10. Interaction log — recent interactions
        try:
            interactions = []
            if hasattr(mind, '_interaction_log'):
                for entry in list(mind._interaction_log)[-30:]:
                    interactions.append(entry)
                interactions.reverse()
            state["interaction_log"] = interactions
        except Exception:
            state["interaction_log"] = []

        # 11. Consciousness self-model
        try:
            consciousness = {}
            if hasattr(mind, 'consciousness'):
                c = mind.consciousness
                if hasattr(c, 'get_state'):
                    consciousness = c.get_state()
                elif hasattr(c, 'self_awareness'):
                    consciousness["self_awareness"] = round(float(c.self_awareness), 3)
            state["consciousness"] = consciousness
        except Exception:
            state["consciousness"] = {}

        # 12. Development phase
        try:
            dev = {}
            if hasattr(mind, 'development'):
                d = mind.development
                if hasattr(d, 'get_state'):
                    ds = d.get_state()
                    dev["phase"] = ds.get("phase", 0)
                    dev["phase_name"] = ds.get("phase_name", "Unknown")
                    dev["age_hours"] = round(ds.get("age_hours", 0), 1)
                    dev["milestones"] = ds.get("milestones_achieved", [])
            state["development"] = dev
        except Exception:
            state["development"] = {}

        return state

    def stop(self):
        """Stop the dashboard."""
        self._running = False

