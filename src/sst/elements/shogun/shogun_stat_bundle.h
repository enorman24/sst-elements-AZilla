
#ifndef _H_SHOGUN_STAT_BUNDLE
#define _H_SHOGUN_STAT_BUNDLE

#include <sst/core/component.h>
#include <sst/core/statapi/statbase.h>

using namespace SST;

namespace SST {
namespace Shogun {

    class ShogunStatisticsBundle {

    public:
        ShogunStatisticsBundle(const int ports)
            : port_count(ports)
        {

            output_packet_count = (Statistic<uint64_t>**)malloc(sizeof(Statistic<uint64_t>*) * port_count);
            input_packet_count = (Statistic<uint64_t>**)malloc(sizeof(Statistic<uint64_t>*) * port_count);
            xbar_stalls = (Statistic<uint64_t>**)malloc(sizeof(Statistic<uint64_t>*) * port_count);
            input_queue_occupancy = (Statistic<uint64_t>**)malloc(sizeof(Statistic<uint64_t>*) * port_count);

            for (int i = 0; i < port_count; ++i) {
                output_packet_count[i] = nullptr;
                input_packet_count[i] = nullptr;
                xbar_stalls[i] = nullptr;
                input_queue_occupancy[i] = nullptr;
            }
        }

        ~ShogunStatisticsBundle()
        {
            free(output_packet_count);
            free(input_packet_count);
            free(xbar_stalls);
            free(input_queue_occupancy);
        }

        void registerStatistics(SST::Component* comp)
        {
            char* subIDName = new char[256];

            for (int i = 0; i < port_count; ++i) {
                sprintf(subIDName, "port%" PRIi32, i);
                output_packet_count[i] = comp->registerStatistic<uint64_t>("output_packet_count", subIDName);
                input_packet_count[i] = comp->registerStatistic<uint64_t>("input_packet_count", subIDName);
                xbar_stalls[i] = comp->registerStatistic<uint64_t>("xbar_stalls", subIDName);
                input_queue_occupancy[i] = comp->registerStatistic<uint64_t>("input_queue_occupancy", subIDName);
            }

            packetsMoved = comp->registerStatistic<uint64_t>("packets_moved");

            delete[] subIDName;
        }

        Statistic<uint64_t>* getOutputPacketCount(const int port)
        {
            return output_packet_count[port];
        }

        Statistic<uint64_t>* getInputPacketCount(const int port)
        {
            return input_packet_count[port];
        }

        Statistic<uint64_t>* getPacketsMoved()
        {
            return packetsMoved;
        }

        Statistic<uint64_t>* getXBarStalls(const int port)
        {
            return xbar_stalls[port];
        }

        Statistic<uint64_t>* getInputQueueOccupancy(const int port)
        {
            return input_queue_occupancy[port];
        }

    private:
        const int port_count;
        Statistic<uint64_t>** output_packet_count;
        Statistic<uint64_t>** input_packet_count;
        Statistic<uint64_t>** xbar_stalls;
        Statistic<uint64_t>** input_queue_occupancy;
        Statistic<uint64_t>* packetsMoved;
    };

}
}

#endif
