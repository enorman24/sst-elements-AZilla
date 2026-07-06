// TeraNoC SST model: the message that rides inside SimpleNetwork::Request
// payloads through the shogun crossbar(s). Mirrors the fields the RTL carries
// in tcdm_master_req_t / tcdm_master_resp_t (addr, amo/wen, meta/core id for
// the return path).

#ifndef _H_TERANOC_EVENT
#define _H_TERANOC_EVENT

#include <sst/core/event.h>

namespace SST {
namespace TeraNoC {

    class TeraNoCMsg : public SST::Event {

    public:
        enum MsgType {
            LOAD_REQ  = 0,
            STORE_REQ = 1,
            AMO_REQ   = 2,
            LOAD_RSP  = 3,
            STORE_ACK = 4
        };

        TeraNoCMsg()
            : type(LOAD_REQ), addr(0), size(4),
              init_group(0), init_tile(0), init_core(0), init_port(0),
              req_id(0) {}

        TeraNoCMsg(MsgType t, uint64_t a, uint32_t sz, uint64_t id, uint16_t core)
            : type(t), addr(a), size(sz),
              init_group(0), init_tile(0), init_core(core), init_port(0),
              req_id(id) {}

        bool isRequest() const { return type == LOAD_REQ || type == STORE_REQ || type == AMO_REQ; }

        SST::Event* clone() override
        {
            return new TeraNoCMsg(*this);
        }

        void serialize_order(SST::Core::Serialization::serializer& ser) override
        {
            Event::serialize_order(ser);
            int t = static_cast<int>(type);
            ser& t;
            type = static_cast<MsgType>(t);
            ser& addr;
            ser& size;
            ser& init_group;
            ser& init_tile;
            ser& init_core;
            ser& init_port;
            ser& req_id;
        }

        ImplementSerializable(SST::TeraNoC::TeraNoCMsg);

    public:
        MsgType  type;
        uint64_t addr;        // global byte address - drives all routing
        uint32_t size;        // bytes (4 for a word)
        uint16_t init_group;  // return-path coordinates of the issuing core
        uint16_t init_tile;
        uint16_t init_core;
        uint16_t init_port;
        uint64_t req_id;      // matches response to an outstanding-request slot
    };

}
}

#endif
