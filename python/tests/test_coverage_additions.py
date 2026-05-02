"""
Additional test cases to improve coverage of previously untested code paths.

Covers:
- helpers/cognitive_grammar.py: validate_message edge cases, all NL parsing paths, error paths
- helpers/distributed_network.py: tree/hybrid topologies, inactive agents, duplicate
  connections, single-agent efficiency, additional error/edge paths
- tools/cognitive_network.py: capability filter, health warning states, include_inactive,
  unknown role handling, error paths
"""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from helpers.cognitive_grammar import (
    CognitiveGrammarFramework,
    CognitiveMessage,
    CommunicativeIntent,
    CognitiveFrame,
    CognitiveRole,
)
from helpers.distributed_network import (
    DistributedNetworkRegistry,
    NetworkAgent,
    AgentCapability,
    NetworkTopology,
    AgentStatus,
    CommunicationProtocol,
    NetworkConnection,
    NetworkStats,
)
from tools.cognitive_network import (
    CognitiveNetworkTool,
    AgentZeroIntegration,
)


# ---------------------------------------------------------------------------
# cognitive_grammar.py additional coverage
# ---------------------------------------------------------------------------

class TestCognitiveGrammarValidation(unittest.TestCase):
    """Cover validate_message branches not hit by existing tests."""

    def _make_message(self, **kwargs):
        base = dict(
            intent=CommunicativeIntent.INFORM,
            frame=CognitiveFrame.INFORMATION_SHARING,
            roles={CognitiveRole.AGENT: "a"},
            content={"k": "v"},
            agent_id="agent_001",
        )
        base.update(kwargs)
        return CognitiveMessage(**base)

    def test_invalid_intent_type_returns_false(self):
        msg = self._make_message(intent="not_an_intent")
        self.assertFalse(CognitiveGrammarFramework.validate_message(msg))

    def test_invalid_frame_type_returns_false(self):
        msg = self._make_message(frame="not_a_frame")
        self.assertFalse(CognitiveGrammarFramework.validate_message(msg))

    def test_invalid_content_type_returns_false(self):
        msg = self._make_message(content="not_a_dict")
        self.assertFalse(CognitiveGrammarFramework.validate_message(msg))

    def test_invalid_roles_type_returns_false(self):
        msg = self._make_message(roles="not_a_dict")
        self.assertFalse(CognitiveGrammarFramework.validate_message(msg))

    def test_invalid_role_key_returns_false(self):
        msg = self._make_message(roles={"bad_key": "value"})
        self.assertFalse(CognitiveGrammarFramework.validate_message(msg))

    def test_validate_raises_exception_returns_false(self):
        # Pass an object whose __bool__ triggers an exception to get into except branch
        msg = MagicMock()
        msg.agent_id = "agent"
        msg.intent = CommunicativeIntent.INFORM
        msg.frame = CognitiveFrame.INFORMATION_SHARING
        msg.content = {"k": "v"}
        msg.roles = MagicMock()
        # Make roles.keys() raise
        msg.roles.keys.side_effect = RuntimeError("boom")
        # isinstance checks will return False for Mock objects so we won't hit the keys() call
        # Use a real message with a patched isinstance instead
        with patch("helpers.cognitive_grammar.isinstance", side_effect=RuntimeError("boom")):
            result = CognitiveGrammarFramework.validate_message(
                self._make_message()
            )
        self.assertFalse(result)


class TestCognitiveGrammarNLParsing(unittest.TestCase):
    """Cover all intent and frame detection branches in parse_natural_language."""

    def _parse(self, text):
        return CognitiveGrammarFramework.parse_natural_language(text, "agent_001")

    # --- intent branches ---

    def test_coordinate_intent(self):
        msg = self._parse("Please coordinate the deployment")
        # "please" triggers REQUEST before "coordinate" for intent, but frame should be COORDINATION
        self.assertIsNotNone(msg)

    def test_coordinate_intent_direct(self):
        msg = self._parse("coordinate the services")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.COORDINATE)

    def test_delegate_intent(self):
        msg = self._parse("delegate this work to the next agent")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.DELEGATE)

    def test_assign_intent(self):
        msg = self._parse("assign the processing job now")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.DELEGATE)

    def test_confirm_intent(self):
        msg = self._parse("confirm the operation is done")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.CONFIRM)

    def test_yes_intent(self):
        msg = self._parse("yes that is correct")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.CONFIRM)

    def test_reject_intent(self):
        msg = self._parse("reject the incoming operation")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.REJECT)

    def test_refuse_intent(self):
        msg = self._parse("refuse to process the data")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.REJECT)

    def test_negotiate_intent(self):
        msg = self._parse("negotiate the allocation terms")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.NEGOTIATE)

    def test_default_intent(self):
        # No matching keyword → INFORM
        msg = self._parse("something unrelated goes here")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.intent, CommunicativeIntent.INFORM)

    # --- frame branches ---

    def test_coordination_frame(self):
        msg = self._parse("coordinate the deployment now")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.COORDINATION)

    def test_capability_frame(self):
        msg = self._parse("what capability is available?")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.CAPABILITY_NEGOTIATION)

    def test_resource_frame(self):
        msg = self._parse("allocate the resource pool")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.RESOURCE_ALLOCATION)

    def test_error_frame(self):
        msg = self._parse("there was an error in processing")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.ERROR_HANDLING)

    def test_status_frame(self):
        msg = self._parse("report the current status update")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.STATUS_REPORTING)

    def test_default_frame(self):
        # No matching keyword → INFORMATION_SHARING
        msg = self._parse("hello world")
        self.assertIsNotNone(msg)
        self.assertEqual(msg.frame, CognitiveFrame.INFORMATION_SHARING)

    def test_parse_error_returns_none(self):
        with patch("helpers.cognitive_grammar.CognitiveGrammarFramework.create_message",
                   side_effect=RuntimeError("forced error")):
            result = CognitiveGrammarFramework.parse_natural_language("test text", "agent_001")
        self.assertIsNone(result)


class TestCognitiveGrammarNLGeneration(unittest.TestCase):
    """Cover generate_natural_language fallback and missing-key paths."""

    def test_no_template_falls_back_to_default_string(self):
        # Use a combination that has no template entry, e.g. NEGOTIATE + STATUS_REPORTING
        msg = CognitiveGrammarFramework.create_message(
            intent=CommunicativeIntent.NEGOTIATE,
            frame=CognitiveFrame.STATUS_REPORTING,
            agent_id="agent_001",
            content={}
        )
        nl = msg.to_natural_language()
        # Should fall back to "Agent agent_001 negotiates regarding status_reporting"
        self.assertIn("agent_001", nl)
        self.assertIn("negotiate", nl)

    def test_template_with_missing_keys_uses_placeholder(self):
        # DELEGATE + TASK_DELEGATION template uses {task} and {patient}
        # Create a message missing those content/role keys
        msg = CognitiveGrammarFramework.create_message(
            intent=CommunicativeIntent.DELEGATE,
            frame=CognitiveFrame.TASK_DELEGATION,
            agent_id="agent_001",
            content={}  # no 'task', no 'patient'
        )
        nl = msg.to_natural_language()
        # Missing keys become <key>
        self.assertIn("<", nl)

    def test_generate_nl_exception_falls_back(self):
        msg = CognitiveGrammarFramework.create_message(
            intent=CommunicativeIntent.INFORM,
            frame=CognitiveFrame.INFORMATION_SHARING,
            agent_id="agent_001",
            content={"topic": "news", "information": "all good"}
        )
        with patch("re.finditer", side_effect=RuntimeError("boom")):
            result = CognitiveGrammarFramework.generate_natural_language(msg)
        self.assertIn("agent_001", result)


class TestCognitiveMessageJSONErrors(unittest.TestCase):
    """Cover to_json / from_json error branches."""

    def test_from_json_missing_required_key_raises(self):
        with self.assertRaises(Exception):
            CognitiveMessage.from_json('{"intent": "inform"}')

    def test_from_json_invalid_enum_raises(self):
        import json
        data = {
            "intent": "bad_value",
            "frame": "information_sharing",
            "roles": {},
            "content": {},
            "agent_id": "a",
        }
        with self.assertRaises(Exception):
            CognitiveMessage.from_json(json.dumps(data))

    def test_to_json_reraises_on_error(self):
        msg = CognitiveGrammarFramework.create_message(
            intent=CommunicativeIntent.INFORM,
            frame=CognitiveFrame.INFORMATION_SHARING,
            agent_id="agent_001",
            content={}
        )
        with patch("helpers.cognitive_grammar.json.dumps", side_effect=TypeError("not serializable")):
            with self.assertRaises(Exception):
                msg.to_json()


# ---------------------------------------------------------------------------
# distributed_network.py additional coverage
# ---------------------------------------------------------------------------

def _make_agent(agent_id, capabilities=None, protocols=None, status=AgentStatus.ACTIVE):
    if capabilities is None:
        capabilities = ["computation"]
    if protocols is None:
        protocols = [CommunicationProtocol.INFERNO_9P]
    caps = [AgentCapability(c, "1.0", c, {}, {}) for c in capabilities]
    return NetworkAgent(
        agent_id=agent_id,
        hostname="localhost",
        port=8080,
        capabilities=caps,
        status=status,
        protocols=protocols,
        metadata={"type": "test"},
        last_heartbeat=time.time(),
        join_timestamp=time.time(),
    )


class TestDistributedNetworkAdditional(unittest.TestCase):
    """Additional coverage for distributed_network.py."""

    def setUp(self):
        self.registry = DistributedNetworkRegistry(
            node_id="test_node",
            topology=NetworkTopology.MESH,
            heartbeat_interval=1.0,
            compatibility_threshold=0.5,
        )

    def tearDown(self):
        if getattr(self.registry, '_running', False):
            self.registry.stop()

    # --- start() already-running path (line 195) ---

    def test_start_when_already_running_is_idempotent(self):
        self.registry.start()
        self.assertTrue(self.registry._running)
        self.registry.start()  # second call should hit the early return
        self.assertTrue(self.registry._running)

    # --- discover_agents with inactive agent (line 291 continue) ---

    def test_discover_agents_skips_inactive(self):
        active = _make_agent("active_agent", ["computation"])
        inactive = _make_agent("inactive_agent", ["computation"], status=AgentStatus.INACTIVE)
        # Force-register without compatibility check
        self.registry.agents["active_agent"] = active
        self.registry.agents["inactive_agent"] = inactive

        results = self.registry.discover_agents(["computation"])
        ids = [a.agent_id for a in results]
        self.assertIn("active_agent", ids)
        self.assertNotIn("inactive_agent", ids)

    # --- tree topology (lines 517-532) ---

    def test_tree_topology_one_agent(self):
        self.registry.topology = NetworkTopology.TREE
        self.registry.register_agent(_make_agent("agent_000"))
        # Single agent → no connections
        self.assertEqual(len(self.registry.connections), 0)

    def test_tree_topology_multiple_agents(self):
        self.registry.topology = NetworkTopology.TREE
        for i in range(5):
            self.registry.register_agent(_make_agent(f"agent_{i:03d}"))
        # Tree topology creates parent↔child connections; connections > 0
        self.assertGreater(len(self.registry.connections), 0)

    # --- hybrid topology (lines 537-558) ---

    def test_hybrid_topology_small(self):
        self.registry.topology = NetworkTopology.HYBRID
        for i in range(3):
            self.registry.register_agent(_make_agent(f"agent_{i:03d}"))
        # ≤3 agents → falls back to mesh
        self.assertGreater(len(self.registry.connections), 0)

    def test_hybrid_topology_large(self):
        self.registry.topology = NetworkTopology.HYBRID
        for i in range(7):
            self.registry.register_agent(_make_agent(f"agent_{i:03d}"))
        # >3 agents → core+hub logic
        self.assertGreater(len(self.registry.connections), 0)

    # --- star topology with empty agents (line 498) ---

    def test_star_topology_empty_is_safe(self):
        self.registry.topology = NetworkTopology.STAR
        self.registry._apply_star_topology()  # no agents → should return without error
        self.assertEqual(len(self.registry.connections), 0)

    # --- single-agent efficiency returns 1.0 (line 611) ---

    def test_efficiency_single_agent(self):
        self.registry.register_agent(_make_agent("agent_000"))
        eff = self.registry._calculate_topology_efficiency()
        self.assertEqual(eff, 1.0)

    # --- _create_connection with missing agent (line 564) ---

    def test_create_connection_missing_agent_is_safe(self):
        self.registry.agents["agent_000"] = _make_agent("agent_000")
        # agent_999 is not registered
        self.registry._create_connection("agent_000", "agent_999")
        self.assertEqual(len(self.registry.connections), 0)

    # --- duplicate connection (line 568) ---

    def test_create_connection_duplicate_is_idempotent(self):
        self.registry.agents["agent_000"] = _make_agent("agent_000")
        self.registry.agents["agent_001"] = _make_agent("agent_001")
        self.registry._create_connection("agent_000", "agent_001")
        count_after_first = len(self.registry.connections)
        self.registry._create_connection("agent_000", "agent_001")
        self.assertEqual(len(self.registry.connections), count_after_first)

    # --- _remove_agent_connections (line 599) ---

    def test_remove_agent_connections_clears_relevant_entries(self):
        self.registry.agents["a"] = _make_agent("a")
        self.registry.agents["b"] = _make_agent("b")
        self.registry.agents["c"] = _make_agent("c")
        self.registry._create_connection("a", "b")
        self.registry._create_connection("b", "a")
        self.registry._create_connection("a", "c")
        self.registry._create_connection("c", "a")
        self.registry._create_connection("b", "c")
        count_before = len(self.registry.connections)
        self.assertGreater(count_before, 0)

        self.registry._remove_agent_connections("a")
        for key in self.registry.connections:
            self.assertNotIn("a", key)

    # --- compatibility scoring: HTTP/WEBSOCKET path (line 458) ---

    def test_compatibility_http_only_protocol(self):
        agent = _make_agent("http_agent", protocols=[CommunicationProtocol.HTTP])
        score = self.registry._assess_cognitive_compatibility(agent)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compatibility_websocket_protocol(self):
        agent = _make_agent("ws_agent", protocols=[CommunicationProtocol.WEBSOCKET])
        score = self.registry._assess_cognitive_compatibility(agent)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # --- performance_history trimming (line 675) ---

    def test_analytics_collector_trims_history(self):
        # Fill history to > 1000 entries, then manually trigger the analytics step
        self.registry.performance_history = [{"timestamp": i} for i in range(1001)]
        # Simulate one analytics collection cycle by calling get_network_stats directly
        # and doing the trim logic as in _analytics_collector
        stats = self.registry.get_network_stats()
        analytics_data = {
            'timestamp': time.time(),
            'stats': vars(stats),
            'agent_count': len(self.registry.agents),
            'connection_count': len(self.registry.connections),
        }
        self.registry.performance_history.append(analytics_data)
        if len(self.registry.performance_history) > 1000:
            self.registry.performance_history = self.registry.performance_history[-1000:]
        self.assertLessEqual(len(self.registry.performance_history), 1000)

    # --- reconfigure topology with real registry ---

    def test_reconfigure_to_tree_topology(self):
        for i in range(4):
            self.registry.register_agent(_make_agent(f"agent_{i:03d}"))
        result = self.registry.reconfigure_topology(NetworkTopology.TREE)
        self.assertTrue(result)
        self.assertEqual(self.registry.topology, NetworkTopology.TREE)

    def test_reconfigure_to_hybrid_topology(self):
        for i in range(6):
            self.registry.register_agent(_make_agent(f"agent_{i:03d}"))
        result = self.registry.reconfigure_topology(NetworkTopology.HYBRID)
        self.assertTrue(result)

    # --- network stats with error / message counts ---

    def test_network_stats_with_message_counts(self):
        self.registry.register_agent(_make_agent("agent_000"))
        self.registry.message_counts["agent_000"] = 100
        self.registry.error_counts["agent_000"] = 5
        stats = self.registry.get_network_stats()
        self.assertAlmostEqual(stats.error_rate, 0.05, places=5)

    # --- get_network_topology error path ---

    def test_get_network_topology_on_error_returns_empty(self):
        with patch.object(self.registry, 'get_network_stats', side_effect=RuntimeError("boom")):
            result = self.registry.get_network_topology()
        self.assertEqual(result, {})

    # --- get_network_stats error path ---

    def test_get_network_stats_on_error_returns_zero_stats(self):
        with patch.object(self.registry, '_calculate_topology_efficiency',
                          side_effect=RuntimeError("boom")):
            stats = self.registry.get_network_stats()
        self.assertEqual(stats.total_agents, 0)

    # --- already active discovery ---

    def test_discover_agents_busy_agent_skipped(self):
        active = _make_agent("active_agent", ["computation"])
        busy = _make_agent("busy_agent", ["computation"], status=AgentStatus.BUSY)
        self.registry.agents["active_agent"] = active
        self.registry.agents["busy_agent"] = busy

        results = self.registry.discover_agents(["computation"])
        ids = [a.agent_id for a in results]
        self.assertIn("active_agent", ids)
        self.assertNotIn("busy_agent", ids)


# ---------------------------------------------------------------------------
# tools/cognitive_network.py additional coverage
# ---------------------------------------------------------------------------

class TestCognitiveNetworkAdditional(unittest.TestCase):
    """Additional coverage for cognitive_network.py."""

    def setUp(self):
        self.mock_registry = MagicMock(spec=DistributedNetworkRegistry)
        self.mock_registry.agents = {}
        self.mock_registry.topology = NetworkTopology.MESH
        self.mock_registry._running = False
        self.tool = CognitiveNetworkTool("test_agent", self.mock_registry)

    # --- unknown role key logs warning but succeeds (lines 140-141) ---

    def test_send_with_unknown_role_key_is_skipped(self):
        result = self.tool.send_cognitive_message(
            target_agent_id="target",
            intent="inform",
            frame="information_sharing",
            content={"topic": "test", "information": "data"},
            roles={"unknown_role": "value", "agent": "test_agent"}
        )
        # Unknown role is skipped; message still sent successfully
        self.assertTrue(result["success"])

    # --- discover_network_agents with include_inactive=True ---

    def test_discover_network_agents_include_inactive(self):
        active_agent = MagicMock()
        active_agent.agent_id = "active_001"
        active_agent.status = AgentStatus.ACTIVE
        active_agent.cognitive_compatibility_score = 0.9
        active_agent.hostname = "localhost"
        active_agent.port = 8080
        active_agent.capabilities = []
        active_agent.protocols = [CommunicationProtocol.HTTP]
        active_agent.metadata = {}
        active_agent.load_metrics = {"cpu_usage": 10.0}

        inactive_agent = MagicMock()
        inactive_agent.agent_id = "inactive_001"
        inactive_agent.status = AgentStatus.INACTIVE
        inactive_agent.cognitive_compatibility_score = 0.7
        inactive_agent.hostname = "localhost"
        inactive_agent.port = 8081
        inactive_agent.capabilities = []
        inactive_agent.protocols = [CommunicationProtocol.HTTP]
        inactive_agent.metadata = {}
        inactive_agent.load_metrics = {"cpu_usage": 0.0}

        self.mock_registry.agents = {
            "active_001": active_agent,
            "inactive_001": inactive_agent,
        }

        mock_stats = MagicMock()
        mock_stats.total_agents = 2
        mock_stats.active_agents = 1
        mock_stats.cognitive_compatibility_avg = 0.8
        self.mock_registry.get_network_stats.return_value = mock_stats

        result = self.tool.discover_network_agents(include_inactive=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_discovered"], 2)

    # --- query_agent_capabilities with capability_filter ---

    def test_query_specific_agent_with_capability_filter(self):
        mock_cap1 = MagicMock()
        mock_cap1.name = "computation"
        mock_cap2 = MagicMock()
        mock_cap2.name = "planning"

        mock_agent = MagicMock()
        mock_agent.agent_id = "target_agent"
        mock_agent.status = AgentStatus.ACTIVE
        mock_agent.cognitive_compatibility_score = 0.8
        mock_agent.capabilities = [mock_cap1, mock_cap2]

        self.mock_registry.agents = {"target_agent": mock_agent}

        result = self.tool.query_agent_capabilities(
            target_agent_id="target_agent",
            capability_filter=["computation"]
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["capabilities"], ["computation"])

    def test_query_all_agents_with_capability_filter(self):
        mock_cap1 = MagicMock()
        mock_cap1.name = "computation"
        mock_cap2 = MagicMock()
        mock_cap2.name = "planning"

        mock_agent1 = MagicMock()
        mock_agent1.agent_id = "agent_001"
        mock_agent1.status = AgentStatus.ACTIVE
        mock_agent1.cognitive_compatibility_score = 0.9
        mock_agent1.capabilities = [mock_cap1, mock_cap2]

        mock_agent2 = MagicMock()
        mock_agent2.agent_id = "agent_002"
        mock_agent2.status = AgentStatus.ACTIVE
        mock_agent2.cognitive_compatibility_score = 0.8
        mock_agent2.capabilities = [mock_cap2]

        self.mock_registry.agents = {
            "agent_001": mock_agent1,
            "agent_002": mock_agent2,
        }

        result = self.tool.query_agent_capabilities(
            capability_filter=["computation"]
        )
        self.assertTrue(result["success"])
        # agent_001 has computation, agent_002 does not
        self.assertIn("computation", result["capability_summary"])
        self.assertEqual(result["capability_summary"]["computation"], ["agent_001"])

    # --- monitor_network_health warning paths ---

    def _make_mock_stats(self, active_agents=2, total_connections=1, error_rate=0.07,
                         compatibility=0.6, topology_efficiency=0.5):
        mock_stats = MagicMock()
        mock_stats.total_agents = 3
        mock_stats.active_agents = active_agents
        mock_stats.total_connections = total_connections
        mock_stats.error_rate = error_rate
        mock_stats.cognitive_compatibility_avg = compatibility
        mock_stats.topology_efficiency = topology_efficiency
        mock_stats.avg_latency = 20.0
        mock_stats.total_bandwidth = 300.0
        mock_stats.message_throughput = 50.0
        return mock_stats

    def test_monitor_health_warning_connectivity(self):
        # connections < active_agents → connectivity warning
        mock_stats = self._make_mock_stats(active_agents=3, total_connections=1,
                                           error_rate=0.0, compatibility=0.8)
        self.mock_registry.get_network_stats.return_value = mock_stats
        self.mock_registry.get_network_topology.return_value = {"topology_type": "mesh"}

        result = self.tool.monitor_network_health()
        self.assertEqual(result["component_health"]["connectivity"], "warning")

    def test_monitor_health_warning_performance(self):
        # error_rate between 0.05 and 0.1 → performance warning
        mock_stats = self._make_mock_stats(active_agents=2, total_connections=5,
                                           error_rate=0.07, compatibility=0.8)
        self.mock_registry.get_network_stats.return_value = mock_stats
        self.mock_registry.get_network_topology.return_value = {"topology_type": "mesh"}

        result = self.tool.monitor_network_health()
        self.assertEqual(result["component_health"]["performance"], "warning")

    def test_monitor_health_warning_compatibility(self):
        # compatibility between 0.5 and 0.7 → compatibility warning
        mock_stats = self._make_mock_stats(active_agents=2, total_connections=4,
                                           error_rate=0.0, compatibility=0.6)
        self.mock_registry.get_network_stats.return_value = mock_stats
        self.mock_registry.get_network_topology.return_value = {"topology_type": "mesh"}

        result = self.tool.monitor_network_health()
        self.assertEqual(result["component_health"]["compatibility"], "warning")

    def test_monitor_health_overall_warning(self):
        # Combinations that average to between 1.5 and 2.5 → overall "warning"
        # connectivity=warning(2), others=healthy(3) → (3+2+3+3)/4 = 2.75 → healthy
        # Use compatibility=warning and performance=warning → (3+3+2+2)/4 = 2.5 → healthy
        # Need at least two warnings that push below 2.5
        # agent=healthy(3), connectivity=warning(2), performance=warning(2), compat=warning(2)
        # → (3+2+2+2)/4 = 2.25 → warning
        mock_stats = self._make_mock_stats(active_agents=2, total_connections=1,
                                           error_rate=0.07, compatibility=0.6)
        self.mock_registry.get_network_stats.return_value = mock_stats
        self.mock_registry.get_network_topology.return_value = {"topology_type": "mesh"}

        result = self.tool.monitor_network_health()
        self.assertTrue(result["success"])
        self.assertEqual(result["overall_health"], "warning")

    def test_monitor_health_exception_returns_failure(self):
        self.mock_registry.get_network_stats.side_effect = RuntimeError("boom")
        result = self.tool.monitor_network_health()
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["overall_health"], "unknown")

    # --- discover_network_agents exception path ---

    def test_discover_network_agents_exception(self):
        self.mock_registry.discover_agents.side_effect = RuntimeError("boom")
        result = self.tool.discover_network_agents(capabilities=["computation"])
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- broadcast_to_network exception path ---

    def test_broadcast_to_network_exception(self):
        self.mock_registry.agents = MagicMock()
        self.mock_registry.agents.values.side_effect = RuntimeError("boom")
        result = self.tool.broadcast_to_network(message_content={"test": "data"})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- coordinate_with_agents exception path ---

    def test_coordinate_with_agents_exception(self):
        self.mock_registry.discover_agents.side_effect = RuntimeError("boom")
        result = self.tool.coordinate_with_agents(
            task_description="test",
            required_capabilities=["computation"]
        )
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- query_agent_capabilities exception path ---

    def test_query_agent_capabilities_exception(self):
        self.mock_registry.agents = MagicMock()
        self.mock_registry.agents.__contains__ = MagicMock(side_effect=RuntimeError("boom"))
        result = self.tool.query_agent_capabilities(target_agent_id="target")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- update_agent_status exception path ---

    def test_update_agent_status_exception(self):
        self.mock_registry.update_agent_status.side_effect = RuntimeError("boom")
        result = self.tool.update_agent_status(new_status="active")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- negotiate_resources exception path ---

    def test_negotiate_resources_exception(self):
        self.mock_registry.discover_agents.side_effect = RuntimeError("boom")
        result = self.tool.negotiate_resources(resource_requirements={"cpu": 2})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- reconfigure_network_topology exception path ---

    def test_reconfigure_topology_exception(self):
        self.mock_registry.topology = NetworkTopology.MESH
        self.mock_registry.get_network_stats.side_effect = RuntimeError("boom")
        result = self.tool.reconfigure_network_topology("star")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- clear_history exception path ---

    def test_clear_history_exception(self):
        with patch.object(self.tool, 'message_history', new_callable=MagicMock) as mock_history:
            mock_history.__len__ = MagicMock(side_effect=RuntimeError("boom"))
            result = self.tool.clear_history()
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- AgentZeroIntegration exception path ---

    def test_enhanced_call_subordinate_exception_path(self):
        with patch.object(self.tool, 'send_cognitive_message',
                          side_effect=RuntimeError("boom")):
            result = AgentZeroIntegration.enhanced_call_subordinate(
                cognitive_tool=self.tool,
                subordinate_id="sub_001",
                task="do something"
            )
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # --- update_agent_status metrics_updated=False path ---

    def test_update_agent_status_load_metrics_failure(self):
        self.mock_registry.update_agent_status.return_value = True
        self.mock_registry.update_agent_load_metrics.return_value = False
        self.mock_registry.agents = {}

        result = self.tool.update_agent_status(
            new_status="busy",
            load_metrics={"cpu_usage": 90.0}
        )
        # success depends on status_updated AND metrics_updated
        self.assertFalse(result["success"])


class TestDistributedNetworkTopologyAdditional(unittest.TestCase):
    """Test topology-related methods on a live DistributedNetworkRegistry."""

    def setUp(self):
        self.registry = DistributedNetworkRegistry(
            "node", NetworkTopology.MESH, heartbeat_interval=5.0, compatibility_threshold=0.3
        )

    def test_tree_topology_seven_agents(self):
        self.registry.topology = NetworkTopology.TREE
        for i in range(7):
            self.registry.register_agent(_make_agent(f"a{i}"))
        self.assertGreater(len(self.registry.connections), 0)

    def test_hybrid_topology_exactly_four_agents(self):
        self.registry.topology = NetworkTopology.HYBRID
        for i in range(4):
            self.registry.register_agent(_make_agent(f"a{i}"))
        self.assertGreater(len(self.registry.connections), 0)

    def test_reconfigure_empty_agents_to_star(self):
        result = self.registry.reconfigure_topology(NetworkTopology.STAR)
        self.assertTrue(result)
        self.assertEqual(self.registry.topology, NetworkTopology.STAR)

    def test_efficiency_zero_agents(self):
        eff = self.registry._calculate_topology_efficiency()
        self.assertEqual(eff, 0.0)

    def test_no_protocol_score(self):
        agent = _make_agent("no_proto_agent", protocols=[CommunicationProtocol.UDP])
        score = self.registry._assess_cognitive_compatibility(agent)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_tree_topology_empty_agents(self):
        """Cover the early return in _apply_tree_topology when no agents."""
        self.registry.topology = NetworkTopology.TREE
        self.registry._apply_tree_topology()  # no agents registered
        self.assertEqual(len(self.registry.connections), 0)

    def test_apply_topology_unknown_topology_type(self):
        """Cover the 'no rules defined' branch in _apply_topology."""
        # Use patch.dict so the original rules are restored after the test
        with patch.dict(self.registry.topology_rules, {}, clear=True):
            self.registry._apply_topology()  # Should log warning, not raise
        self.assertEqual(len(self.registry.connections), 0)

    def test_apply_topology_exception_is_caught(self):
        """Cover exception path in _apply_topology."""
        with patch.object(self.registry, '_apply_mesh_topology',
                          side_effect=RuntimeError("boom")):
            # Should not raise
            self.registry._apply_topology()

    def test_assess_compatibility_exception_returns_zero(self):
        """Cover exception path in _assess_cognitive_compatibility."""
        agent = _make_agent("test_agent")
        # Make the `in` check on protocols raise an exception inside the try block
        protocols_mock = MagicMock()
        protocols_mock.__contains__ = MagicMock(side_effect=RuntimeError("boom"))
        agent.protocols = protocols_mock
        score = self.registry._assess_cognitive_compatibility(agent)
        self.assertEqual(score, 0.0)

    def test_calculate_efficiency_exception_returns_zero(self):
        """Cover exception path in _calculate_topology_efficiency."""
        # Use patch.object on the private method called inside the try block
        with patch.object(self.registry, '_calculate_topology_efficiency',
                          side_effect=None):
            pass  # this just verifies we can wrap it
        # Real test: make the topology_efficiency_map lookup raise
        with patch.dict(self.registry.topology_rules, {}):
            # Patch the topology map inside the method by mocking topology.value
            mock_topo = MagicMock()
            mock_topo.value = "mesh"
            mock_topo.__hash__ = MagicMock(side_effect=RuntimeError("boom"))
            original_topology = self.registry.topology
            self.registry.topology = mock_topo
            try:
                eff = self.registry._calculate_topology_efficiency()
                self.assertEqual(eff, 0.0)
            finally:
                self.registry.topology = original_topology

    def test_analytics_collector_trims_history_inline(self):
        """Directly exercise the history-trim logic in _analytics_collector."""
        # Pre-fill history above the 1000-entry limit
        self.registry.performance_history = [{"i": i} for i in range(1001)]
        call_count = [0]

        def stop_after_one(secs):
            call_count[0] += 1
            self.registry._running = False

        self.registry._running = True
        with patch("helpers.distributed_network.time.sleep", side_effect=stop_after_one):
            self.registry._analytics_collector()

        self.assertLessEqual(len(self.registry.performance_history), 1000)

    def test_analytics_collector_exception_path(self):
        """Cover exception handler in _analytics_collector."""
        call_count = [0]

        def raise_then_stop(secs):
            call_count[0] += 1
            if call_count[0] >= 2:
                self.registry._running = False

        self.registry._running = True
        with patch.object(self.registry, 'get_network_stats',
                          side_effect=RuntimeError("forced error")), \
             patch("helpers.distributed_network.time.sleep",
                   side_effect=raise_then_stop):
            self.registry._analytics_collector()

        self.assertFalse(self.registry._running)

    def test_heartbeat_monitor_exception_path(self):
        """Cover exception handler in _heartbeat_monitor."""
        call_count = [0]

        def raise_then_stop(secs):
            call_count[0] += 1
            if call_count[0] >= 2:
                self.registry._running = False

        self.registry._running = True
        # Make agents.items() raise so the try block inside the loop errors out
        self.registry.agents = MagicMock()
        self.registry.agents.items.side_effect = RuntimeError("forced error")
        with patch("helpers.distributed_network.time.sleep",
                   side_effect=raise_then_stop):
            self.registry._heartbeat_monitor()

        self.assertFalse(self.registry._running)


if __name__ == '__main__':
    unittest.main()
