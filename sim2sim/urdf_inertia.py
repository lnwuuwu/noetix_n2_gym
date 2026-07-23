"""Synchronize MJCF rigid-body inertias with the URDF used for training."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def _vector(value, expected_size=3):
    result = np.fromstring(value or "0 0 0", sep=" ", dtype=np.float64)
    if result.size != expected_size:
        raise ValueError(
            "Expected {} values, received {!r}".format(expected_size, value)
        )
    return result


def _rotation_from_rpy(value):
    roll, pitch, yaw = _vector(value)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _link_inertial(link, body_rotation, body_translation):
    inertial = link.find("inertial")
    if inertial is None:
        return None
    origin = inertial.find("origin")
    local_position = _vector(
        None if origin is None else origin.attrib.get("xyz")
    )
    local_rotation = _rotation_from_rpy(
        None if origin is None else origin.attrib.get("rpy")
    )
    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None:
        raise ValueError(
            "Incomplete URDF inertia for link '{}'".format(link.attrib["name"])
        )
    mass = float(mass_element.attrib["value"])
    values = {
        name: float(inertia_element.attrib[name])
        for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    }
    inertia = np.asarray(
        [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ],
        dtype=np.float64,
    )
    rotation = body_rotation @ local_rotation
    return (
        mass,
        body_translation + body_rotation @ local_position,
        rotation @ inertia @ rotation.T,
    )


def _combine_inertials(components, body_name):
    if not components:
        raise ValueError("No URDF inertia found for MJCF body: " + body_name)
    total_mass = sum(component[0] for component in components)
    if total_mass <= 0.0:
        raise ValueError("Non-positive URDF mass for MJCF body: " + body_name)
    center = sum(
        mass * position for mass, position, _ in components
    ) / total_mass
    inertia = np.zeros((3, 3), dtype=np.float64)
    for mass, position, component_inertia in components:
        displacement = position - center
        inertia += component_inertia + mass * (
            np.dot(displacement, displacement) * np.eye(3)
            - np.outer(displacement, displacement)
        )
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(
            "URDF inertia for '{}' is not positive definite: {}".format(
                body_name, eigenvalues
            )
        )
    return total_mass, center, inertia


def _format_vector(values):
    return " ".join("{:.17g}".format(float(value)) for value in values)


def align_mjcf_inertials_from_urdf(mjcf_root, urdf_path):
    """Replace MJCF inertials with URDF values, merging omitted fixed links.

    The repository's historical MJCF predates several URDF inertia changes.
    MuJoCo also collapses the two fixed hand links into their elbow bodies, so
    their mass, center of mass, and parallel-axis contribution are included in
    the parent inertial here.
    """
    urdf_root = ET.parse(urdf_path).getroot()
    links = {
        link.attrib["name"]: link for link in urdf_root.findall("link")
    }
    mjcf_bodies = {
        body.attrib["name"]: body
        for body in mjcf_root.findall("./worldbody//body")
        if "name" in body.attrib
    }
    fixed_children = {}
    for joint in urdf_root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError("Fixed URDF joint is missing parent/child")
        origin = joint.find("origin")
        fixed_children.setdefault(parent.attrib["link"], []).append(
            (
                child.attrib["link"],
                _rotation_from_rpy(
                    None if origin is None else origin.attrib.get("rpy")
                ),
                _vector(
                    None if origin is None else origin.attrib.get("xyz")
                ),
            )
        )

    def collect(link_name, rotation, translation):
        if link_name not in links:
            raise ValueError("URDF link not found: " + link_name)
        components = []
        component = _link_inertial(
            links[link_name], rotation, translation
        )
        if component is not None:
            components.append(component)
        for child_name, child_rotation, child_translation in fixed_children.get(
            link_name, ()
        ):
            if child_name in mjcf_bodies:
                continue
            components.extend(
                collect(
                    child_name,
                    rotation @ child_rotation,
                    translation + rotation @ child_translation,
                )
            )
        return components

    updated = []
    for body_name, body in mjcf_bodies.items():
        if body_name not in links:
            continue
        mass, center, inertia = _combine_inertials(
            collect(body_name, np.eye(3), np.zeros(3)), body_name
        )
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.Element("inertial")
            body.insert(0, inertial)
        inertial.attrib.pop("quat", None)
        inertial.attrib.pop("diaginertia", None)
        inertial.set("pos", _format_vector(center))
        inertial.set("mass", "{:.17g}".format(mass))
        inertial.set(
            "fullinertia",
            _format_vector(
                (
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                )
            ),
        )
        updated.append(body_name)

    missing = sorted(set(links) - set(mjcf_bodies))
    fixed_descendants = {
        child
        for children in fixed_children.values()
        for child, _, _ in children
    }
    unexpected = [name for name in missing if name not in fixed_descendants]
    if unexpected:
        raise ValueError(
            "URDF links have no MJCF bodies and are not fixed children: "
            + ", ".join(unexpected)
        )
    return tuple(updated)
