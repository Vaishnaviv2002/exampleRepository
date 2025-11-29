#!/usr/bin/env python3
"""
Export trained PyBrain network (pickle) into an XML file
called UE_05_App3_ANN_Model.xml.
"""

import sys
import pickle
import xml.etree.ElementTree as ET

sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.module import Module

PKL_IN = "pybrain_trained_net_dataset03.pkl"
XML_OUT = "UE_05_App3_ANN_Model.xml"

def export_network_to_xml(net, out_path):
    root = ET.Element("FeedForwardNetwork")
    root.set("modules", str(len(net.modules)))

    # document modules (layers)
    modules_el = ET.SubElement(root, "Modules")
    for m in net.modules:
        mod_el = ET.SubElement(modules_el, "Module")
        mod_el.set("name", m.name)
        mod_el.set("size", str(m.outdim))
        mod_el.set("type", m.__class__.__name__)
        mod_el.set("bias", str(getattr(m, "bias", False)))

    # document connections (weights)
    connections_el = ET.SubElement(root, "Connections")
    for conn in net.connections.values():
        for c in conn:
            conn_el = ET.SubElement(connections_el, "Connection")
            conn_el.set("from", c.inmod.name)
            conn_el.set("to", c.outmod.name)
            conn_el.set("type", c.__class__.__name__)

            # write weight matrix
            w_el = ET.SubElement(conn_el, "Weights")
            for i, row in enumerate(c.params.reshape(c.outdim, c.indim)):
                row_el = ET.SubElement(w_el, "Row")
                row_el.set("index", str(i))
                row_el.text = " ".join(f"{v:.10f}" for v in row)

    # pretty print + save
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print("Saved ANN model as XML:", out_path)

def main():
    print("Loading trained network:", PKL_IN)
    with open(PKL_IN, "rb") as f:
        net = pickle.load(f)

    export_network_to_xml(net, XML_OUT)

if __name__ == "__main__":
    main()
