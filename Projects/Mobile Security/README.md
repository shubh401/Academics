# Mobile Security - Winter, 2018-19 (Course Instructor - Dr.-Ing. Sven Bugiel)

#### This course project included two individual components which targeted security extensions for inter-process communication (*IPC*) in Android Framework to track call-chains of the intents as well as to define additional *SELinux* policies and enforce fine-grained mandatory access-control when applications access privileged resources in the system. 

We refer the reader to have a look at the project report for a detailed description of the problem statement, the concept as well as the proposed solution and its implementation. Broadly, the project consists of two components:

1. IPC Call-Chain Tracking for provenance information required by the reference monitor.

2. Additional SELinux policy extension in the Android middleware to enforce fine-grained mandatory-access control over installed applications.