"""
Privilege Graph Construction for SentinelIQ.
Builds NetworkX graph of user-role-system relationships.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class PrivilegeGraph:
    """Builds and analyzes privilege relationship graphs."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, users_df: pd.DataFrame, events_df: pd.DataFrame) -> nx.DiGraph:
        """
        Build privilege graph from identity data.
        
        Nodes: Users, Roles, Systems, Resources
        Edges: Access relationships with weights.
        """
        self.graph = nx.DiGraph()

        # Add user nodes
        for _, user in users_df.iterrows():
            self.graph.add_node(
                user["user_id"],
                node_type="user",
                label=user["username"],
                department=user["department"],
                privilege_level=user["privilege_level"],
                risk_color=self._get_risk_color(user.get("privilege_level", "user")),
            )

            # Add system nodes and edges
            systems = str(user.get("systems_access", "")).split("|")
            for system in systems:
                if system and system.strip():
                    system = system.strip()
                    if not self.graph.has_node(system):
                        self.graph.add_node(
                            system,
                            node_type="system",
                            label=system,
                            sensitivity=self._get_system_sensitivity(system),
                        )
                    self.graph.add_edge(
                        user["user_id"],
                        system,
                        relationship="has_access",
                        privilege=user["privilege_level"],
                    )

            # Add department node
            dept = user["department"]
            dept_node = f"dept_{dept}"
            if not self.graph.has_node(dept_node):
                self.graph.add_node(dept_node, node_type="department", label=dept)
            self.graph.add_edge(user["user_id"], dept_node, relationship="belongs_to")

        # Add event-based edges (resource access)
        if not events_df.empty:
            resource_access = events_df.groupby(["user_id", "resource"]).size().reset_index(name="access_count")
            for _, row in resource_access.iterrows():
                resource_node = f"res_{row['resource']}"
                if not self.graph.has_node(resource_node):
                    self.graph.add_node(resource_node, node_type="resource", label=row["resource"])
                self.graph.add_edge(
                    row["user_id"],
                    resource_node,
                    relationship="accessed",
                    weight=row["access_count"],
                )

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def get_blast_radius(self, user_id: str) -> Dict:
        """
        Calculate blast radius for a compromised account.
        What systems/resources can this account access?
        
        Args:
            user_id: User to analyze.
            
        Returns:
            Dictionary with blast radius analysis.
        """
        if user_id not in self.graph:
            return {"error": f"User {user_id} not found in graph"}

        # Get all reachable nodes from this user
        reachable = nx.descendants(self.graph, user_id)

        systems = []
        resources = []
        departments = []

        for node in reachable:
            node_data = self.graph.nodes[node]
            node_type = node_data.get("node_type", "unknown")
            if node_type == "system":
                systems.append(node)
            elif node_type == "resource":
                resources.append(node_data.get("label", node))
            elif node_type == "department":
                departments.append(node_data.get("label", node))

        # Calculate blast radius score
        blast_score = (
            len(systems) * 10 +
            len(resources) * 5 +
            len(departments) * 3
        )

        return {
            "user_id": user_id,
            "blast_radius_score": min(100, blast_score),
            "systems_at_risk": systems,
            "resources_at_risk": resources,
            "departments_affected": departments,
            "total_reachable_nodes": len(reachable),
            "risk_assessment": (
                "CRITICAL" if blast_score > 50 else
                "HIGH" if blast_score > 30 else
                "MEDIUM" if blast_score > 15 else "LOW"
            ),
        }

    def get_graph_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "user_nodes": len([n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "user"]),
            "system_nodes": len([n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "system"]),
            "resource_nodes": len([n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "resource"]),
            "avg_connections": self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
        }

    def export_for_visualization(self) -> Dict:
        """Export graph data for front-end visualization."""
        nodes = []
        edges = []

        for node, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("label", node),
                "type": data.get("node_type", "unknown"),
                "group": data.get("node_type", "unknown"),
            })

        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "from": source,
                "to": target,
                "relationship": data.get("relationship", ""),
                "weight": data.get("weight", 1),
            })

        return {"nodes": nodes, "edges": edges}

    def _get_risk_color(self, privilege_level: str) -> str:
        """Map privilege level to color."""
        colors = {
            "admin": "#FF0000",
            "power-user": "#FF8C00",
            "service-account": "#FFD700",
            "user": "#00CC00",
        }
        return colors.get(privilege_level, "#808080")

    def _get_system_sensitivity(self, system: str) -> str:
        """Map system to sensitivity level."""
        high_sensitivity = {"PROD_DB", "ADMIN_SYS", "SIEM", "AWS_IAM", "GCP"}
        medium_sensitivity = {"Azure_AD", "Okta", "ServiceNow", "Salesforce"}
        if system in high_sensitivity:
            return "high"
        elif system in medium_sensitivity:
            return "medium"
        return "low"
